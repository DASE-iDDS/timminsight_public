"""Full SemBench experiment on REAL data (paper Section 6).

No sampling. Runs TiMMInsight (qwen-vl) over the ENTIRE labeled image set of each
medical scenario and scores against the curated ground-truth labels that ship
with the SemBench data. Concurrency keeps the full run tractable.

Usage:
  export DASHSCOPE_API_KEY=...
  python3 -m src.experiments.sembench_full --scenario skin
  python3 -m src.experiments.sembench_full --scenario xray
"""

import os
import base64
import glob
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# Data roots are configurable via env; defaults are repo-relative (no data ships here).
_SB = os.environ.get("SEMBENCH_DATA_ROOT", "data/sembench")
ROOT = f"{_SB}/medical/raw_data"
SCEN = {
    "skin": {
        "dir": f"{ROOT}/all_skin_images",
        "glob": "*.jpg",
        "gold": lambda b: 1 if "malignant" in b else (0 if "benign" in b else None),
        "instruction": "This is a dermatology image of a skin lesion. Is it malignant (skin cancer)? Answer strictly YES or NO.",
        "pos": "malignant", "neg": "benign",
    },
    "xray": {
        "dir": f"{ROOT}/all_x_rays",
        "glob": "*.jp*g",
        "gold": lambda b: 0 if "00_normal" in b else (1 if any(f"_0{i}" in b for i in range(1, 9)) else None),
        "instruction": "This is a chest x-ray. Does it show any abnormality or disease (not a normal healthy chest)? Answer strictly YES or NO.",
        "pos": "abnormal", "neg": "normal",
    },
    # ecomm gold comes from styles.csv (gender), images by id. Binary Men vs Women.
    "ecomm": {
        "csv": f"{_SB}/ecomm/fashion-dataset/styles.csv",
        "imgdir": f"{_SB}/ecomm/fashion-dataset/images",
        "instruction": "This is a fashion product image. Is this item for women (rather than men)? Answer strictly YES or NO.",
        "pos": "women", "neg": "men",
    },
    # cars gold = damage_status in image_data_full.csv. Binary damaged vs no_damage.
    "cars": {
        "csv": f"{_SB}/cars/image_data_full.csv",
        "imgdir": f"{_SB}/cars/all_car_images",
        "instruction": "This is a photo related to a car. Does it show vehicle damage such as dents, scratches, torn or lost parts? Answer strictly YES or NO.",
        "pos": "damaged", "neg": "no_damage",
    },
}


def _cars_files(cfg):
    import csv
    out = []
    with open(cfg["csv"], newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            ds = (row.get("damage_status") or "").strip().lower()
            gold = 0 if ds in ("", "none", "no_damage") else 1
            base = os.path.basename(row.get("image_path", ""))
            p = os.path.join(cfg["imgdir"], base)
            if base and os.path.exists(p):
                out.append((p, gold))
    return out


def _ecomm_files(cfg):
    import csv
    out = []
    with open(cfg["csv"], newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            g = row.get("gender", "")
            gold = 1 if g == "Women" else (0 if g == "Men" else None)
            if gold is None:
                continue
            p = os.path.join(cfg["imgdir"], f"{row['id']}.jpg")
            if os.path.exists(p):
                out.append((p, gold))
    return out

_local = threading.local()


def _client():
    if not hasattr(_local, "c"):
        _local.c = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                          base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    return _local.c


def _b64(p):
    return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()


def _predict(path, instruction, retries=4):
    import time
    for attempt in range(retries):
        try:
            r = _client().chat.completions.create(
                model="qwen-vl-max", temperature=0, max_tokens=6,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": _b64(path)}}]}])
            return 1 if r.choices[0].message.content.strip().upper().startswith("Y") else 0
        except Exception:
            time.sleep(2 * (attempt + 1))   # backoff on rate limit / timeout
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCEN))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = full set (no sampling)")
    args = ap.parse_args()
    cfg = SCEN[args.scenario]

    if args.scenario == "ecomm":
        files = _ecomm_files(cfg)
    elif args.scenario == "cars":
        files = _cars_files(cfg)
    else:
        files = [(f, cfg["gold"](os.path.basename(f))) for f in glob.glob(f"{cfg['dir']}/{cfg['glob']}")
                 if not os.path.basename(f).startswith("._")]
        files = [(f, g) for f, g in files if g is not None]
    if args.limit:
        files = files[: args.limit]
    npos = sum(g for _, g in files)
    print(f"FULL SemBench medical/{args.scenario}: {len(files)} labeled images "
          f"({npos} {cfg['pos']}, {len(files)-npos} {cfg['neg']}), workers={args.workers}\n", flush=True)

    tp = fp = tn = fn = done = err = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_predict, f, cfg["instruction"]): g for f, g in files}
        for fu in as_completed(futs):
            g = futs[fu]; pred = fu.result()
            with lock:
                done += 1
                if pred is None:
                    err += 1
                elif g and pred:
                    tp += 1
                elif g and not pred:
                    fn += 1
                elif not g and pred:
                    fp += 1
                else:
                    tn += 1
                if done % 200 == 0:
                    print(f"  ...{done}/{len(files)}", flush=True)

    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0
    P = tp / (tp + fp) if tp + fp else 0
    R = tp / (tp + fn) if tp + fn else 0
    F = 2 * P * R / (P + R) if P + R else 0
    print(f"\n=== FULL SemBench medical/{args.scenario} (n={n}, errors={err}) ===")
    print(f"  TiMMInsight (qwen-vl): accuracy={acc:.3f} precision={P:.3f} recall={R:.3f} F1={F:.3f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")


if __name__ == "__main__":
    main()
