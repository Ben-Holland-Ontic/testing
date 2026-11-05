"""Batched .ocr(list) must equal per-crop .ocr(img), and must be faster."""
import os, sys, time, warnings, glob
warnings.filterwarnings("ignore")
BK = r"C:\OCULI.pd1\oculi-runtime-1.2.7-gpu\bk-main"
for p in (BK, os.path.join(BK,"workers"), os.path.join(BK,"input"), os.path.join(BK,"cleaners")):
    sys.path.insert(0, p)
os.environ.setdefault("PADDLEOCR_USE_GPU","1")
import cv2, numpy as np, cluster_ocr
from rectangle_detection import detect_rectangles

eng = cluster_ocr._get_paddleocr()
crops = []
for f in sorted(glob.glob(r"C:\Users\benja\OneDrive\Desktop\data\*")):
    img = cv2.imread(f)
    if img is None: continue
    try:
        rd, _ = detect_rectangles(image=img, frame_rect=None, page_id=f, adaptive_thresh=True,
                                  debug=False, basename=None,
                                  gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    except Exception as e:
        print("detect failed", f, e); continue
    for r in (rd or {}).get("rectangles", [])[:40]:
        c = r.get("crop")
        if c is not None and getattr(c,"size",0):

            crops.append(c)
print(f"crops: {len(crops)}")
if not crops: sys.exit("no crops")

def norm(res):
    return [[(t, round(float(sc),4)) for _, (t, sc) in items] for items in res]

t0=time.perf_counter(); per=[eng.ocr(c)[0] for c in crops]; t_per=time.perf_counter()-t0
t0=time.perf_counter(); bat=eng.ocr(crops);                 t_bat=time.perf_counter()-t0

a, b = norm([[i] for i in []] ) , None
A = [[(t, round(float(sc),4)) for _, (t, sc) in items] for items in per]
B = [[(t, round(float(sc),4)) for _, (t, sc) in items] for items in bat]
print(f"per-crop : {t_per:6.2f}s  ({len(A)} results)")
print(f"batched  : {t_bat:6.2f}s  ({len(B)} results)   speedup x{t_per/max(t_bat,1e-9):.2f}")
if len(A)!=len(B):
    print("FAIL: length mismatch"); sys.exit(1)
diff = [i for i,(x,y) in enumerate(zip(A,B)) if x!=y]
la = sum(len(x) for x in A); lb = sum(len(x) for x in B)
print(f"lines    : per-crop {la}   batched {lb}")
print(f"identical: {len(A)-len(diff)}/{len(A)}")
for i in diff[:5]:
    print(f"  crop {i}:\n    per={A[i][:3]}\n    bat={B[i][:3]}")
print("PARITY OK" if not diff else f"{len(diff)} DIFFS")

TA = [[t for t,_ in x] for x in A]; TB = [[t for t,_ in x] for x in B]
td = [i for i,(x,y) in enumerate(zip(TA,TB)) if x!=y]
print()
print(f"TEXT-ONLY identical: {len(TA)-len(td)}/{len(TA)}")
mx = max((abs(sa-sb) for x,y in zip(A,B) for (_,sa),(_,sb) in zip(x,y)), default=0)
print(f"max confidence delta: {mx:.6f}")
for i in td[:5]: print(f"  crop {i}: per={TA[i]}  bat={TB[i]}")
