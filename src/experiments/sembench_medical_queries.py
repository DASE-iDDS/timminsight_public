"""Faithful SemBench medical query experiment (paper Section 6).

Runs the ACTUAL SemBench medical queries (Q1, Q3, Q8, Q9) end-to-end the way
SemBench defines them. The system sees only the visible patient table (no
diagnosis labels) plus the raw modalities (symptom text, x-ray images, skin
images) and must decide the semantic predicates with an LLM (text) and a
vision model (images). The answer patient-set is scored against the official
SemBench ground_truth, and reported alongside the published baseline F1 so the
two are directly comparable.

Audio queries (Q2, Q5) are excluded (paper scope is image+text+table).

Usage:
  export DASHSCOPE_API_KEY=...
  python3 -m src.experiments.sembench_medical_queries --queries Q1,Q3,Q8,Q9
"""

import os
import csv
import json
import base64
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# Data roots are configurable via env; defaults are repo-relative (no data ships here).
_SB = os.environ.get("SEMBENCH_DATA_ROOT", "data/sembench")
MED = f"{_SB}/medical/data"
IMG_ROOT = f"{MED}/raw_data"
GT = f"{_SB}/SemBench-main/files/medical/raw_results/ground_truth"
METRICS = f"{_SB}/SemBench-main/files/medical/metrics"

_local = threading.local()


def _client():
    if not hasattr(_local, "c"):
        _local.c = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                          base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    return _local.c


def _csv(path, key=None):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8", errors="ignore")))
    return {r[key]: r for r in rows} if key else rows


def _img(path):
    base = os.path.basename(path)
    for sub in ("all_x_rays", "all_skin_images"):
        p = f"{IMG_ROOT}/{sub}/{base}"
        if os.path.exists(p):
            return p
    return None


def _b64(p):
    return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()


def _text_yes(instruction, value, retries=3):
    for a in range(retries):
        try:
            r = _client().chat.completions.create(model="qwen3.6-plus", temperature=0, max_tokens=6,
                extra_body={"enable_thinking": False},
                messages=[{"role": "system", "content": "Answer strictly YES or NO."},
                          {"role": "user", "content": f"{instruction}\nPatient symptoms: {value}"}])
            return r.choices[0].message.content.strip().upper().startswith("Y")
        except Exception:
            import time; time.sleep(2 * (a + 1))
    return False


def _vis_yes(instruction, img_path, retries=3):
    p = _img(img_path)
    if not p:
        return False
    for a in range(retries):
        try:
            r = _client().chat.completions.create(model="qwen-vl-max", temperature=0, max_tokens=6,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": f"Answer strictly YES or NO. {instruction}"},
                    {"type": "image_url", "image_url": {"url": _b64(p)}}]}])
            return r.choices[0].message.content.strip().upper().startswith("Y")
        except Exception:
            import time; time.sleep(2 * (a + 1))
    return False


def _parallel(items, fn, workers):
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    return out


def _prf(got, gold):
    got, gold = set(got), set(gold)
    tp = len(got & gold); fp = len(got - gold); fn = len(gold - got)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def gold_ids(q):
    rows = _csv(f"{GT}/{q}.csv")
    return {r.get("patient_id") for r in rows if r.get("patient_id")}


def baseline_f1(q):
    out = {}
    for s in ("lotus", "palimpzest", "thalamusdb", "bigquery", "flockmtl"):
        f = f"{METRICS}/{s}.json"
        if os.path.exists(f):
            d = json.load(open(f))
            if q in d and isinstance(d[q], dict):
                out[s] = d[q].get("f1_score")
    return out


def run_query(q, workers):
    patients = _csv(f"{MED}/patient_data.csv", key="patient_id")          # visible structured table
    symptoms = {r["patient_id"]: r["symptoms"] for r in _csv(f"{MED}/text_symptoms_data.csv")}
    xray = {r["patient_id"]: r["image_path"] for r in _csv(f"{MED}/image_x_ray_data.csv")}
    skin = {r["patient_id"]: r["image_path"] for r in _csv(f"{MED}/image_skin_data.csv")}
    gold = gold_ids(q)

    if q == "Q1":   # patients whose symptoms indicate allergy (text semantic filter)
        ids = list(symptoms)
        res = _parallel(ids, lambda i: _text_yes(
            "Do these symptoms indicate an allergy?", symptoms[i]), workers)
        got = {i for i in ids if res.get(i)}
    elif q == "Q3":  # family cancer = 1 AND x-ray shows lung problem (struct + vision)
        ids = [i for i, r in patients.items() if r.get("did_family_have_cancer") == "1" and i in xray]
        res = _parallel(ids, lambda i: _vis_yes(
            "This is a chest x-ray. Does it show a lung problem or abnormality (not normal)?", xray[i]), workers)
        got = {i for i in ids if res.get(i)}
    elif q == "Q8":  # malignant skin moles AND family cancer (vision + struct)
        ids = [i for i, r in patients.items() if r.get("did_family_have_cancer") == "1" and i in skin]
        res = _parallel(ids, lambda i: _vis_yes(
            "This is a skin mole image. Is the mole malignant (skin cancer)?", skin[i]), workers)
        got = {i for i in ids if res.get(i)}
    elif q == "Q9":  # sick by skin AND sick by x-ray (vision AND vision)
        ids = [i for i in patients if i in skin and i in xray]
        rs = _parallel(ids, lambda i: _vis_yes(
            "This is a skin mole image. Is the mole malignant (skin cancer)?", skin[i]), workers)
        rx = _parallel(ids, lambda i: _vis_yes(
            "This is a chest x-ray. Does it show a lung problem or abnormality (not normal)?", xray[i]), workers)
        got = {i for i in ids if rs.get(i) and rx.get(i)}
    else:
        raise ValueError(q)

    p, r, f = _prf(got, gold)
    return {"q": q, "candidates": len(ids), "got": len(got), "gold": len(gold),
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "baselines": baseline_f1(q)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="Q1,Q3,Q8,Q9")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    print(f"Faithful SemBench medical queries: {args.queries}, workers={args.workers}\n", flush=True)
    rows = []
    for q in args.queries.split(","):
        r = run_query(q, args.workers)
        rows.append(r)
        bl = "  ".join(f"{s[:5]}={v:.2f}" for s, v in r["baselines"].items() if v is not None)
        print(f"[{q}] cand={r['candidates']:4d} got={r['got']:4d} gold={r['gold']:4d}  "
              f"TiMMInsight F1={r['f1']:.3f} (P={r['precision']:.2f} R={r['recall']:.2f})  | baselines: {bl}",
              flush=True)
    json.dump(rows, open("runs_tiqc/sembench_medical_queries.json", "w"), indent=2)
    print("\nsaved -> runs_tiqc/sembench_medical_queries.json")


if __name__ == "__main__":
    main()
