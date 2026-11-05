"""Phase 0b smoke test: does Phi-4-mini INT4 run on the VALIDATED onnxruntime?

Answers three questions with measurements, not assertions:

  1. Does onnxruntime-genai 0.9.2 load the model against onnxruntime-gpu 1.22.0 --
     the version already validated and shipping -- without an upgrade?
  2. Does it bind CUDA (not silently fall back to CPU)? Provider fallback is the exact
     failure mode that cost a day on the OCR side: everything works, everything is slow,
     nothing says why.
  3. What does it actually cost on this card -- VRAM resident, tokens/sec, TTFT?

    $PY = 'C:\OCULI.pd1\TESTING\.venv-genai122\Scripts\python.exe'
    & $PY TESTING\RESOURCES\phi_smoke_probe.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

MODEL = Path(r"C:\OCULI.pd1\TESTING\models-llm\phi4-mini-int4\gpu\gpu-int4-rtn-block-32")


def register_cuda_dll_dirs() -> int:
    """Mirror ocr_backend.register_cuda_dll_dirs() for this throwaway venv.

    onnxruntime pulls its CUDA dependencies (cublasLt64_12.dll and friends) from the
    nvidia-* wheels, whose DLL directories must be registered BEFORE onnxruntime loads its
    provider. Production does this in ocr_backend; this venv has no nvidia wheels of its
    own, so borrow the ones from the validated environment. Without it the CUDA provider
    fails to load and genai falls back to CPU -- the same silent-degradation failure this
    probe exists to detect.
    """
    import os
    if os.name != "nt":
        return 0
    added = 0
    for sp in (r"C:\OCULI.pd1\TESTING\.venv-nopaddle\Lib\site-packages",
               os.path.join(sys.prefix, "Lib", "site-packages")):
        nvidia = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia):
            continue
        for pkg in sorted(os.listdir(nvidia)):
            for sub in ("bin", "lib"):
                d = os.path.join(nvidia, pkg, sub)
                if os.path.isdir(d):
                    try:
                        os.add_dll_directory(d)
                        added += 1
                    except Exception:
                        pass
    return added


def gpu_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def main() -> int:
    n_dlls = register_cuda_dll_dirs()   # MUST precede any onnxruntime import
    print(f"cuda dll dirs        {n_dlls} registered")
    import onnxruntime as ort
    print(f"onnxruntime          {ort.__version__}")
    import onnxruntime_genai as og
    print(f"onnxruntime-genai    {og.__version__}")

    if ort.__version__ != "1.22.0":
        print(f"  !! expected the VALIDATED 1.22.0, got {ort.__version__}")

    if not MODEL.is_dir():
        print(f"model dir missing: {MODEL}")
        return 2

    cfg = MODEL / "genai_config.json"
    if cfg.is_file():
        c = json.loads(cfg.read_text(encoding="utf-8"))
        provs = (c.get("model", {}).get("decoder", {})
                  .get("session_options", {}).get("provider_options", []))
        print(f"declared providers   {[list(p)[0] for p in provs] or 'none (CPU)'}")

    base = gpu_mib()
    print(f"\nGPU before load      {base} MiB")

    t0 = time.perf_counter()
    model = og.Model(str(MODEL))
    load_s = time.perf_counter() - t0
    after = gpu_mib()
    print(f"load                 {load_s:.1f}s")
    print(f"GPU after load       {after} MiB   (delta {after - base:+d} MiB)")
    if after - base < 500:
        print("  !! delta is small -- the model may be on CPU, not GPU")

    tokenizer = og.Tokenizer(model)

    def run(prompt: str, max_new: int, label: str):
        """One generation. Returns (n_tokens, seconds, ttft, text)."""
        tokens = tokenizer.encode(prompt)
        params = og.GeneratorParams(model)
        try:
            params.set_search_options(max_length=len(tokens) + max_new, do_sample=False)
        except Exception:
            params.set_search_options(max_length=len(tokens) + max_new)
        gen = og.Generator(model, params)
        try:
            gen.append_tokens(tokens)
        except AttributeError:
            params.input_ids = tokens
            gen = og.Generator(model, params)
        stream = tokenizer.create_stream()
        out, n, ttft = [], 0, None
        t0 = time.perf_counter()
        while not gen.is_done():
            gen.generate_next_token()
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
            try:
                out.append(stream.decode(gen.get_next_tokens()[0]))
            except Exception:
                pass
        return n, time.perf_counter() - t0, ttft, "".join(out).strip(), len(tokens)

    # THE VOCABULARY THE DATA-CLEANING STAGE ACTUALLY USES.
    #
    # Not the 10-class LayoutML detector set. BORDER_FRAME / TITLE_BLOCK /
    # REVISION_TABLE / MATRIX_TABLE belong to the detector's training vocabulary; the
    # runtime classifier in data_cleaningv2.py emits only these six (plus ABSTAIN), and
    # frames are resolved earlier by frame_detection, a separate pipeline stage.
    #
    # Offering BORDER_FRAME here is a category error and it cost a real miss: with the
    # 10-class list the model classified 'REV DATE APPROVED DESCRIPTION' as BORDER_FRAME.
    # Constraining the candidate set to what the stage can actually mean is not prompt
    # tuning -- it is removing wrong answers from the option set.
    CLASSES = "TABLE, INFO_CLUSTER, LIST, PARAGRAPH, STRING, KEY_VALUE"

    def classify_prompt(text: str) -> str:
        return ("<|system|>You output only JSON.<|end|><|user|>A region of an engineering "
                f"drawing contains this text:\n{text}\n\nClassify it as exactly one of: "
                f"{CLASSES}.\nReply as {{\"class\": \"...\"}} and nothing else."
                "<|end|><|assistant|>")

    # WARM-UP: the first call pays kernel selection and cuDNN autotune. Measuring
    # throughput on it (as the first version of this probe did) reports the warm-up,
    # not the steady state -- it made GPU look SLOWER than CPU.
    run(classify_prompt("REV DATE APPROVED"), 16, "warmup")

    print("\n--- steady-state throughput (post warm-up) ---")
    long_prompt = ("<|system|>You are precise.<|end|><|user|>Explain, in detail, how a "
                   "bill of materials table is structured.<|end|><|assistant|>")
    n, secs, ttft, _, n_in = run(long_prompt, 256, "long")
    print(f"prompt {n_in} tok -> generated {n} tok in {secs:.2f}s")
    print(f"TTFT                 {ttft * 1000:.0f} ms")
    print(f"THROUGHPUT           {n / secs:.1f} tok/s")
    peak = gpu_mib()
    print(f"GPU peak             {peak} MiB   (delta {peak - base:+d} MiB)")

    print("\n--- classification sanity (the actual Placement 1 task) ---")
    cases = [
        ("REV  DATE  APPROVED  DESCRIPTION", "TABLE"),
        ("DRAWN BY: J.SMITH   CHECKED: A.JONES   DWG NO: 540-1149   SCALE: 1:2",
         "INFO_CLUSTER"),
        ("ITEM  PART NO  QTY  DESCRIPTION\n1  AAA-111  2  BRACKET\n2  BBB-222  1  SPACER",
         "TABLE"),
        ("1. Remove the cover.  2. Inspect the seal.  3. Replace if worn.", "LIST"),
        ("MATERIAL: 6061-T6 ALUMINIUM", "KEY_VALUE"),
        ("This assembly shall be inspected in accordance with the applicable quality "
         "standard before shipment, and any nonconforming item shall be quarantined.",
         "PARAGRAPH"),
        ("540-1202-3", "STRING"),
    ]
    hits = 0
    for text, expected in cases:
        _, s, _, out, _ = run(classify_prompt(text), 24, "cls")
        ok = expected in out
        hits += ok
        print(f"  {'OK ' if ok else 'MISS'} {s * 1000:6.0f} ms  expected {expected:<15} got {out[:60]!r}")
    print(f"  {hits}/{len(cases)} correct  (indicative only -- real eval is the COCO set)")

    print("\n--- verdict ---")
    print(f"runs on validated ORT 1.22.0 : {ort.__version__ == '1.22.0'}")
    print(f"GPU resident                 : {after - base > 500}  ({after - base} MiB after load)")
    print(f"steady-state throughput      : {n / secs:.1f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
