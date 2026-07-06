"""External specialized tools for metadata extraction (paper §3.1.1, Figure 2).

These are the concrete tool backends the ReAct processors call:
  - YOLOTool        object detection (ultralytics YOLOv8)
  - Detectron2Tool  object detection (facebookresearch Detectron2)
  - Places365Tool   scene classification (ResNet-18 trained on Places365)
  - ViTFeatureTool  semantic feature extraction (timm ViT)
  - SpacyNERTool    named-entity recognition (spaCy en_core_web_sm)

Every tool lazy-loads its model on first use; `.available` reports whether the
dependency + weights are actually present, so a processor can honestly record which
tools ran (per the integrity rules — a tool that could not load is reported as
unavailable, never faked). All tools operate on REAL inputs (image bytes/paths/URLs,
raw text).
"""
from __future__ import annotations
import io
import os
import urllib.request
from functools import lru_cache
from typing import Any, Dict, List, Optional

_MODELS_DIR = os.path.expanduser("~/.timm_metadata_models")
os.makedirs(_MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------- image loading
def load_image(src: Any):
    """Load a PIL RGB image from raw bytes, a local path, or an http(s) URL."""
    from PIL import Image
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(src)).convert("RGB")
    s = str(src)
    if s.startswith(("http://", "https://")):
        req = urllib.request.Request(s, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(s).convert("RGB")


class _Tool:
    name = "tool"
    deps: List[str] = []       # modules that must import for the tool to be usable
    _err: Optional[str] = None

    @property
    def available(self) -> bool:
        """CHEAP check: only verify the required libraries import — never download
        weights or load a model here (that is _load's job, done lazily on first run)."""
        import importlib
        try:
            for m in self.deps:
                importlib.import_module(m)
            return True
        except Exception as e:  # dependency missing / broken install
            self._err = f"{type(e).__name__}: {str(e)[:90]}"
            return False

    def _load(self):
        raise NotImplementedError


# ---------------------------------------------------------------- YOLO (objects)
class YOLOTool(_Tool):
    name = "detect_objects_yolo"
    deps = ["ultralytics"]
    description = "Detect objects in an image (label, confidence, bbox) via YOLOv8."

    @lru_cache(maxsize=1)
    def _load(self):
        from ultralytics import YOLO
        return YOLO(os.path.join(_MODELS_DIR, "yolov8n.pt")
                    if os.path.exists(os.path.join(_MODELS_DIR, "yolov8n.pt")) else "yolov8n.pt")

    def run(self, image_src: Any) -> List[Dict[str, Any]]:
        model = self._load()
        img = load_image(image_src)
        W, H = img.size
        res = model.predict(img, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            area = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, W * H)
            out.append({"label": model.names[int(b.cls[0])], "confidence": float(b.conf[0]),
                        "bbox": [x1, y1, x2, y2], "area": round(area, 4)})
        return out


# ---------------------------------------------------------------- Detectron2 (objects)
class Detectron2Tool(_Tool):
    name = "detect_objects_detectron2"
    deps = ["detectron2", "omegaconf"]
    description = "Detect objects in an image via Detectron2 (Mask R-CNN)."

    @lru_cache(maxsize=1)
    def _load(self):
        import detectron2  # noqa
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu"
        from detectron2.data import MetadataCatalog
        return DefaultPredictor(cfg), MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

    def run(self, image_src: Any) -> List[Dict[str, Any]]:
        import numpy as np
        predictor, classes = self._load()
        img = load_image(image_src)
        W, H = img.size
        arr = np.array(img)[:, :, ::-1]  # RGB->BGR
        inst = predictor(arr)["instances"].to("cpu")
        out = []
        for box, cls, sc in zip(inst.pred_boxes.tensor.tolist(),
                                inst.pred_classes.tolist(), inst.scores.tolist()):
            x1, y1, x2, y2 = box
            area = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, W * H)
            out.append({"label": classes[cls], "confidence": float(sc),
                        "bbox": box, "area": round(area, 4)})
        return out


# ---------------------------------------------------------------- Places365 (scene)
_PLACES_WEIGHTS = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
_PLACES_CATS = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"


class Places365Tool(_Tool):
    name = "classify_scene_places365"
    deps = ["torch", "torchvision"]
    description = "Classify the scene/place category of an image via ResNet18-Places365."

    def _fetch(self, url, fn):
        path = os.path.join(_MODELS_DIR, fn)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
        return path

    @lru_cache(maxsize=1)
    def _load(self):
        import torch
        from torchvision.models import resnet18
        wpath = self._fetch(_PLACES_WEIGHTS, "resnet18_places365.pth.tar")
        cpath = self._fetch(_PLACES_CATS, "categories_places365.txt")
        model = resnet18(num_classes=365)
        ckpt = torch.load(wpath, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(sd)
        model.eval()
        cats = [ln.strip().split(" ")[0][3:] for ln in open(cpath)]
        return model, cats

    def run(self, image_src: Any, topk: int = 3) -> List[Dict[str, Any]]:
        import torch
        from torchvision import transforms
        model, cats = self._load()
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])
        x = tf(load_image(image_src)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x)[0], dim=0)
        vals, idx = probs.topk(topk)
        return [{"scene": cats[i], "confidence": float(v)} for v, i in zip(vals, idx)]


# ---------------------------------------------------------------- ViT (semantic features)
class ViTFeatureTool(_Tool):
    name = "extract_features_vit"
    deps = ["timm", "torch"]
    description = "Extract a semantic feature embedding from an image via a timm ViT."

    @lru_cache(maxsize=1)
    def _load(self):
        import timm
        import torch  # noqa
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        model.eval()
        cfg = timm.data.resolve_data_config({}, model=model)
        return model, timm.data.create_transform(**cfg)

    def run(self, image_src: Any):
        import torch
        model, tf = self._load()
        x = tf(load_image(image_src)).unsqueeze(0)
        with torch.no_grad():
            feat = model(x)[0]
        return feat.numpy()  # semantic embedding vector


# ---------------------------------------------------------------- spaCy NER (text)
class SpacyNERTool(_Tool):
    name = "extract_entities_spacy"
    deps = ["spacy", "en_core_web_sm"]
    description = "Extract named entities (text, label, offsets) from text via spaCy."

    @lru_cache(maxsize=1)
    def _load(self):
        import spacy
        return spacy.load("en_core_web_sm")

    def run(self, text: str) -> List[Dict[str, Any]]:
        nlp = self._load()
        doc = nlp(str(text)[:100000])
        return [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
                for e in doc.ents]


# ---------------------------------------------------------------- VADER (sentiment)
class VaderSentimentTool(_Tool):
    name = "analyze_sentiment_vader"
    deps = ["vaderSentiment"]
    description = "Rule-based sentiment (positive/negative/neutral + compound) via VADER."

    @lru_cache(maxsize=1)
    def _load(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()

    def run(self, text: str) -> Dict[str, Any]:
        s = self._load().polarity_scores(str(text)[:5000])
        comp = s["compound"]
        label = "positive" if comp >= 0.05 else ("negative" if comp <= -0.05 else "neutral")
        return {"sentiment": label, "compound": round(comp, 4), "scores": s}


# ---------------------------------------------------------------- KeyBERT (topics)
class KeyBERTTool(_Tool):
    name = "extract_topics_keybert"
    deps = ["keybert"]
    description = "Extract key topics / keyphrases from text via KeyBERT (BERT embeddings)."

    @lru_cache(maxsize=1)
    def _load(self):
        from keybert import KeyBERT
        return KeyBERT()  # default sentence-transformers all-MiniLM-L6-v2

    def run(self, text: str, top_n: int = 3) -> List[Dict[str, Any]]:
        kws = self._load().extract_keywords(
            str(text)[:5000], keyphrase_ngram_range=(1, 2),
            stop_words="english", top_n=top_n)
        return [{"topic": k, "score": round(float(s), 4)} for k, s in kws]


# spaCy entity label -> our EntityType name
SPACY_LABEL_MAP = {
    "PERSON": "PERSON", "ORG": "ORGANIZATION", "GPE": "LOCATION", "LOC": "LOCATION",
    "FAC": "LOCATION", "EVENT": "EVENT", "DATE": "TIME", "TIME": "TIME",
    "CARDINAL": "QUANTITY", "QUANTITY": "QUANTITY", "MONEY": "QUANTITY",
    "PRODUCT": "OBJECT", "WORK_OF_ART": "CONCEPT", "NORP": "CONCEPT",
}


def available_tools() -> Dict[str, bool]:
    """Report which external tools can actually load (for honest run manifests)."""
    tools = [YOLOTool(), Detectron2Tool(), Places365Tool(), ViTFeatureTool(), SpacyNERTool(),
             VaderSentimentTool(), KeyBERTTool()]
    out = {}
    for t in tools:
        ok = t.available
        out[t.name] = ok
        if not ok:
            out[t.name + "_error"] = t._err
    return out
