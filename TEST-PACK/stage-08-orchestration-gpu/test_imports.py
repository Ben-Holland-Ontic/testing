#!/usr/bin/env python3
"""Phase 0 import test for Blackwell dependency tree."""
import sys

def test_import(module_name, attr=None):
    try:
        mod = __import__(module_name, fromlist=[attr] if attr else [])
        if attr:
            val = getattr(mod, attr)
            return True, f"{module_name}.{attr} = {val}"
        return True, f"{module_name} imported OK"
    except Exception as e:
        return False, f"{module_name} FAILED: {e}"

results = []

# Core numeric
for mod in ["numpy", "scipy", "pandas", "sklearn"]:
    ok, msg = test_import(mod)
    results.append((ok, msg))

# Image/OCR base
for mod in ["cv2", "PIL", "skimage", "imgaug", "shapely", "pyclipper", "rapidfuzz"]:
    ok, msg = test_import(mod)
    results.append((ok, msg))

# Paddle
ok, msg = test_import("paddle", "__version__")
results.append((ok, msg))
ok, msg = test_import("paddle", "is_compiled_with_cuda")
results.append((ok, msg))

# PaddleOCR
ok, msg = test_import("paddleocr")
results.append((ok, msg))

# Torch
ok, msg = test_import("torch", "__version__")
results.append((ok, msg))
ok, msg = test_import("torchvision")
results.append((ok, msg))

# DocTR
ok, msg = test_import("doctr")
results.append((ok, msg))

# NLP
for mod in ["spacy", "transformers", "sentence_transformers"]:
    ok, msg = test_import(mod)
    results.append((ok, msg))

# PDF
for mod in ["pdfplumber", "pypdfium2", "fitz"]:
    ok, msg = test_import(mod)
    results.append((ok, msg))

# Print summary
print("=" * 60)
print("IMPORT TEST RESULTS")
print("=" * 60)
all_ok = True
for ok, msg in results:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {msg}")
    if not ok:
        all_ok = False
print("=" * 60)
if all_ok:
    print("ALL IMPORTS PASSED")
else:
    print("SOME IMPORTS FAILED")
    sys.exit(1)
