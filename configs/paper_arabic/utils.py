import re
import base64
import io
from pathlib import Path
from PIL import Image

import jiwer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from loguru import logger
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

# ---------------- metrics helpers ------------------------
def normalize_text(t):
    if isinstance(t, list):
        t = " ".join(t)
    if t is None:
        return ""
    t = t.replace("\n", " ")
    return re.sub(r"\s+", " ", t).strip()

def levenshtein(a: str, b: str) -> int:
    _, n = len(a), len(b)
    dp = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(
                dp[j] + 1,      # del
                dp[j - 1] + 1,  # ins
                prev + (ca != cb)  # sub
            )
    return dp[n]

def cer(ref: str, hyp: str) -> float:
    if not ref.strip():
        return 1.0
    dist = levenshtein(ref, hyp)
    return dist / max(1, len(ref))

def bleu4(ref: str, hyp: str) -> float:
    if not ref.strip():
        return 0.0
    return sentence_bleu([list(ref)], list(hyp),
                         smoothing_function=SmoothingFunction().method4)

def wer(ref: str, hyp: str) -> float:
    if not ref.strip():
        return 1.0
    return jiwer.wer(ref, hyp)


# ---------------- image helpers ------------------------
def _load_image(obj):
    """
    Universal image loading.
    Supports:
        - PIL.Image -> returns RGB
        - path (str / Path) -> opens via PIL
        - bytes -> loads from bytes
        - data URI (base64) -> decodes
    """
    if obj is None:
        return None

    if isinstance(obj, Image.Image):
        return obj.convert("RGB")

    if isinstance(obj, (Path,)):
        obj = str(obj)

    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("data:image"):
            s = re.sub(r"^data:image/[^;]+;base64,", "", s)
            return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
        try:
            return Image.open(s).convert("RGB")
        except Exception:
            try:
                return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
            except Exception:
                raise
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(obj)}")


# ---------------- ref-finding helpers ------------------------
def find_ref_for_image(img_path, entries):
    """
    Iterate through entries; if img_path is present in e.get("images", []),
    return the content of the message with role == "assistant".
    """
    if not entries:
        return None
    for e in entries:
        imgs = e.get("images", []) or []
        if img_path in imgs:
            for m in e.get("messages", []) or []:
                if m.get("role") == "assistant":
                    return m.get("content")
    return None


def find_ref_in_doc(doc, global_entries=None):
    """
    Attempts to find a reference string in doc:
     1) doc.get("response") or doc.get("ref")
     2) if doc contains a messages structure -> search for a message with role == "assistant"
     3) if doc contains image_path (or image) and global_entries is provided -> call find_ref_for_image
    """
    if "response" in doc and doc["response"] is not None:
        return doc["response"]
    if "ref" in doc and doc["ref"] is not None:
        return doc["ref"]
    if "text" in doc and doc["text"] is not None:
        return doc["text"]

    messages = doc.get("messages") or []
    if messages:
        for m in messages:
            if m.get("role") == "assistant" and m.get("content"):
                return m.get("content")

    img_key_candidates = ["image_path", "image", "images"]
    img_val = None
    for k in img_key_candidates:
        if k in doc and doc[k]:
            img_val = doc[k]
            break

    if img_val and global_entries:
        if isinstance(img_val, (list, tuple)):
            for ip in img_val:
                found = find_ref_for_image(ip, global_entries)
                if found:
                    return found
        else:
            return find_ref_for_image(img_val, global_entries)

    return None


# ---------------- lmms_eval entrypoints ------------------------
def paper_arabic_to_visual(doc):
    """
    Returns a list of images. If doc contains an "image" field (PIL object or path),
    or "image_path", or "images" — process them.
    """
    imgs = []
    if "image" in doc and doc["image"] is not None:
        try:
            imgs.append(_load_image(doc["image"]))
        except Exception:
            try:
                imgs.append(_load_image(doc.get("image_path") or doc.get("image")))
            except Exception:
                imgs = []
    elif "image_path" in doc and doc["image_path"]:
        try:
            imgs.append(_load_image(doc["image_path"]))
        except Exception:
            imgs = []
    elif "images" in doc and doc["images"]:
        for it in doc["images"]:
            try:
                imgs.append(_load_image(it))
            except Exception:
                continue
    return imgs


def paper_arabic_doc_to_text(doc, lmms_eval_specific_kwargs):
    return lmms_eval_specific_kwargs.get("prompt", "")


def paper_arabic_process_results(doc, results, global_entries=None):
    """
    Similar to the original loop:
    - results[0] is treated as the prediction (string)
    - attempt to find a reference for the image
    - if reference is missing -> mark the entry as skipped
    """
    pred_raw = results[0] if results else ""
    pred = normalize_text(pred_raw)

    ref_raw = find_ref_in_doc(doc, global_entries)
    if ref_raw is None:
        return {
            "paper_arabic_metrics": {
                "ref": "",
                "pred": pred,
                "WER": None,
                "CER": None,
                "BLEU-4": None,
                "image": doc.get("image_path") or doc.get("id") or doc.get("image"),
                "proc_time": 0.0,
                "skipped_empty_ref": 1,
            }
        }

    ref = normalize_text(ref_raw)


    if not ref.strip():
        return {
            "paper_arabic_metrics": {
                "ref": ref,
                "pred": pred,
                "WER": None,
                "CER": None,
                "BLEU-4": None,
                "image": doc.get("image_path") or doc.get("id") or doc.get("image"),
                "proc_time": 0.0,
                "skipped_empty_ref": 1,
            }
        }

    w = wer(ref, pred)
    c = cer(ref, pred)
    b = bleu4(ref, pred)

    return {
        "paper_arabic_metrics": {
            "ref": ref,
            "pred": pred,
            "WER": w,
            "CER": c,
            "BLEU-4": b,
            "image": doc.get("image_path") or doc.get("id") or doc.get("image"),
            "skipped_empty_ref": 0,
        }
    }


def paper_arabic_aggregate_metrics(results, args):
    """
    Aggregate metrics and auxiliary counters. Return a dictionary with detailed statistics,
    and also write a file similar to the original.
    Expected format of results: a list of elements like:
      { "ref":..., "pred":..., "WER":..., "CER":..., "BLEU-4":..., "skipped_empty_ref": 0/1 }
    """
    records = []
    skipped_empty_ref = 0

    wer_list, cer_list, bleu_list = [], [], []

    for item in results:
        if isinstance(item, dict) and "paper_arabic_metrics" in item:
            m = item["paper_arabic_metrics"]
        else:
            m = item

        skipped_empty_ref += int(m.get("skipped_empty_ref", 0))

        if m.get("WER") is not None:
            wer_list.append(m["WER"])
        if m.get("CER") is not None:
            cer_list.append(m["CER"])
        if m.get("BLEU-4") is not None:
            bleu_list.append(m["BLEU-4"])

        records.append({
            "image": m.get("image"),
            "ref": m.get("ref"),
            "hyp": m.get("pred"),
            "WER": m.get("WER"),
            "CER": m.get("CER"),
            "BLEU-4": m.get("BLEU-4"),
        })

    processed = len(records)
    valid = processed - skipped_empty_ref

    avg_wer = sum(wer_list) / len(wer_list) if wer_list else 0.0
    avg_cer = sum(cer_list) / len(cer_list) if cer_list else 0.0
    avg_bleu = sum(bleu_list) / len(bleu_list) if bleu_list else 0.0
    
    file_name = generate_submission_file("paper_arabic_metrics.txt", args, subpath="results")
    with open(file_name, "w", encoding="utf-8") as f:
        print("#################### Metrics for paper ####################", file=f)
        print(f"Processed entries (seen by process_results): {processed}", file=f)
        print(f"Skipped entries with empty refs:            {skipped_empty_ref}", file=f)
        print(f"Valid (processed - skipped):               {valid}", file=f)
        if valid:
            print(f"Average WER:  {avg_wer:.4f}", file=f)
            print(f"Average CER:  {avg_cer:.4f}", file=f)
            print(f"Average BLEU: {avg_bleu:.4f}", file=f)
        else:
            print("No valid records to compute WER/CER/BLEU.", file=f)

    logger.info(f"Metrics for paper saved to {file_name}")

    return {
        "processed": processed,
        "skipped_empty_ref": skipped_empty_ref,
        "valid": valid,
        "avg_WER": avg_wer,
        "avg_CER": avg_cer,
        "avg_BLEU": avg_bleu,
    }