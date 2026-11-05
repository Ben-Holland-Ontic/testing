"""Vendored oculi_ocr vs the paddleocr/paddlex oracle — the gate for removing the packages.

DIAGNOSTIC. Both paths use the same ONNX models and the same GPU, so the ONLY variable is
the pre/post-processing implementation. Anything that differs is a porting error.

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY vendored_parity_probe.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

BK = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main"
for p in (BK, os.path.join(BK, "workers")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ort_gpu_guard import register_cuda_dll_dirs  # noqa: E402

register_cuda_dll_dirs()

import types  # noqa: E402
for nm in ("modelscope", "modelscope.hub", "modelscope.utils"):
    sys.modules.setdefault(nm, types.ModuleType(nm))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

ONNX = os.path.join(BK, "models", "paddleocr", "onnx")
DET = os.path.join(ONNX, "PP-OCRv6_medium_det_infer")
REC = os.path.join(ONNX, "PP-OCRv6_medium_rec_infer")
CLS = os.path.join(ONNX, "PP-LCNet_x0_25_textline")
IMAGES = [os.path.join(BK, "test_data", "01_original.png"),
          os.path.join(BK, "test_data", "Screenshot_21-7-2026_101813_.jpeg")]

# Exactly cluster_ocr's tuning.
KW = dict(text_det_limit_side_len=960, text_det_limit_type="max",
          text_det_box_thresh=0.0, text_det_thresh=0.1, text_det_unclip_ratio=2.0,
          text_rec_score_thresh=0.0)


def oracle(img_path):
    from paddleocr import PaddleOCR
    from paddlex.inference.pipelines import load_pipeline_config
    import json
    cfg = json.loads(json.dumps(load_pipeline_config("OCR")))
    for name in ("TextDetection", "TextRecognition", "TextLineOrientation"):
        sm = cfg.setdefault("SubModules", {}).setdefault(name, {})
        sm["engine"] = "onnxruntime"
        ec = dict(sm.get("engine_config") or {})
        ec["onnxruntime"] = {"providers": ["CUDAExecutionProvider"]}
        sm["engine_config"] = ec
    ocr = PaddleOCR(paddlex_config=cfg,
                    text_detection_model_name="PP-OCRv6_medium_det",
                    text_detection_model_dir=DET,
                    text_recognition_model_name="PP-OCRv6_medium_rec",
                    text_recognition_model_dir=REC,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=True,
                    textline_orientation_model_name="PP-LCNet_x0_25_textline_ori",
                    textline_orientation_model_dir=CLS, device="gpu:0",
                    return_word_box=False, **KW)
    t0 = time.perf_counter()
    r = ocr.predict(img_path)[0]
    dt = time.perf_counter() - t0
    d = r if isinstance(r, dict) else r.json.get("res", {})
    return list(d.get("rec_texts", [])), list(d.get("rec_scores", [])), \
        [np.asarray(b) for b in d.get("rec_polys", d.get("dt_polys", []))], dt


def vendored(img_path):
    from oculi_ocr import OculiOCR
    ocr = OculiOCR(text_detection_model_dir=DET, text_recognition_model_dir=REC,
                   use_gpu=True, use_textline_orientation=True,
                   textline_orientation_model_dir=CLS, **KW)
    t0 = time.perf_counter()
    r = ocr.predict(img_path)[0]
    dt = time.perf_counter() - t0
    return list(r["rec_texts"]), list(r["rec_scores"]), \
        [np.asarray(b) for b in r["rec_polys"]], dt


def centre(b):
    return np.asarray(b, dtype=np.float64).reshape(-1, 2).mean(axis=0)


def main() -> int:
    ok_all = True
    for img in IMAGES:
        if not os.path.isfile(img):
            continue
        print("=" * 76)
        print(os.path.basename(img))
        print("=" * 76)
        ot, os_, ob, odt = oracle(img)
        vt, vs, vb, vdt = vendored(img)
        print(f"  oracle  (paddleocr): {len(ot):4d} texts  {odt:5.2f}s")
        print(f"  vendored (oculi_ocr): {len(vt):4d} texts  {vdt:5.2f}s")

        # Match on box centre, never list index: if the two disagree on one box every
        # later index shifts and one real difference reads as hundreds.
        used, pairs = set(), []
        for i, b in enumerate(ob):
            c = centre(b)
            best, bd = None, 1e18
            for j, b2 in enumerate(vb):
                if j in used:
                    continue
                dd = float(np.hypot(*(centre(b2) - c)))
                if dd < bd:
                    best, bd = j, dd
            if best is not None and bd <= 6.0:
                used.add(best)
                pairs.append((i, best, bd))

        same = sum(1 for i, j, _ in pairs if ot[i] == vt[j])
        offs = [d for _, _, d in pairs] or [0.0]
        print(f"  boxes matched : {len(pairs)}/{max(len(ob), len(vb))}  "
              f"centre offset mean {sum(offs)/len(offs):.3f}px max {max(offs):.3f}px")
        print(f"  text identical: {same}/{len(pairs)} "
              f"({same / max(len(pairs),1) * 100:.2f}%)")
        if pairs:
            ds = [abs(os_[i] - vs[j]) for i, j, _ in pairs
                  if i < len(os_) and j < len(vs)]
            if ds:
                print(f"  score delta   : mean {sum(ds)/len(ds):.6f} max {max(ds):.6f}")
        shown = 0
        for i, j, _ in pairs:
            if ot[i] != vt[j] and shown < 10:
                print(f"    oracle={ot[i]!r}  vendored={vt[j]!r}")
                shown += 1
        page_ok = (len(ot) == len(vt) and len(pairs) == len(ot) and same == len(pairs))
        ok_all &= page_ok
        print(f"  -> {'MATCH' if page_ok else 'DIFFERS'}")

    print("\n" + "=" * 76)
    print("VERDICT:", "VENDORED IMPLEMENTATION MATCHES THE ORACLE"
          if ok_all else "DIFFERENCES — porting error, inspect above")
    print("=" * 76)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
