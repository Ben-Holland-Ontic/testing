"""Phase 1 - input adapters and decoders for the four supporting recognisers.

The ONNX conversions were verified numerically against the reference implementations,
but only on RANDOM tensors. The preprocessing and the decoding in
`workers/ocr_supporting.py` are new code that no conversion check covered, and they are
exactly where a silent wrong answer comes from: a missing inversion, a resize with the
wrong filter, an off-by-one in a charset. So the roundtrip test compares against what
kraken / strhub / doctr actually read on the same crops
(RESOURCES/supporting_reference_probe.py records those).

    $PY = 'C:\\OCULI.pd1\\TESTING\\.venv-nopaddle\\Scripts\\python.exe'
    & $PY TESTING\\tests\\test_phase1_adapters.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import add_runtime_to_path, check, check_eq, main  # noqa: E402

# The models are not packaged into the runtime tree yet (that is a Phase 4/7 decision),
# so point the loader at the conversion output.
os.environ.setdefault("OCULI_SUPPORTING_MODEL_DIR", r"C:\OCULI.pd1\TESTING\onnx-models")
os.environ.setdefault("PADDLEOCR_USE_GPU", "1")

add_runtime_to_path()

FIX = Path(r"C:\OCULI.pd1\TESTING\RESOURCES\fixtures")
CROPS = FIX / "crops"
MANIFEST = FIX / "crops.json"
REFERENCE = FIX / "reference.json"


def _load_crops(limit=None):
    import cv2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if limit:
        manifest = manifest[:limit]
    out = []
    for m in manifest:
        img = cv2.imread(str(CROPS / m["file"]))
        if img is not None:
            out.append((m, img))
    return out


def test_adapter_shapes():
    """Each adapter produces exactly the tensor its ONNX graph declares."""
    import numpy as np
    import ocr_supporting as S

    crops = [img for _, img in _load_crops(limit=5)]
    if not check("fixture crops present", bool(crops), f"{len(crops)} loaded"):
        return

    checks = [
        ("catmus", S.catmus_batch, (len(crops), 1, 120, None)),
        ("parseq", S.parseq_batch, (len(crops), 3, 32, 128)),
        ("sar", S.doctr_batch, (len(crops), 3, 32, 128)),
    ]
    for name, fn, want in checks:
        x = fn(crops)
        check_eq(f"{name}: dtype", str(x.dtype), "float32")
        got = tuple(x.shape)
        ok = len(got) == 4 and all(w is None or g == w for g, w in zip(got, want))
        check(f"{name}: shape {got}", ok, "" if ok else f"want {want}")

    # ...and the graph agrees with the adapter, rather than the adapter agreeing with
    # my memory of the graph.
    import onnxruntime as ort
    for name, fn in (("catmus", S.catmus_batch), ("parseq", S.parseq_batch),
                     ("sar", S.doctr_batch), ("master", S.doctr_batch)):
        onnx_path, _ = S.model_paths(name)
        if not os.path.isfile(onnx_path):
            check(f"{name}: model file present", False, onnx_path)
            continue
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        declared = sess.get_inputs()[0].shape
        x = fn(crops)
        ok = len(declared) == x.ndim and all(
            isinstance(d, str) or d == s for d, s in zip(declared, x.shape))
        check(f"{name}: adapter matches graph input {declared}", ok,
              "" if ok else f"adapter gave {x.shape}")

    # CATMuS width is dynamic: two batches of different crops must differ in width, or
    # the adapter is squashing to a fixed size and losing long lines -- the one thing
    # CATMuS is in the cascade to avoid.
    wide = S.catmus_batch([c for c in crops if c.shape[1] > 140] or crops)
    narrow = S.catmus_batch([min(crops, key=lambda c: c.shape[1] / c.shape[0])])
    check("catmus: width really is dynamic", wide.shape[3] != narrow.shape[3],
          f"{wide.shape[3]} vs {narrow.shape[3]}")

    # Every value finite -- a NaN here would silently poison a whole batch.
    for name, fn in (("catmus", S.catmus_batch), ("parseq", S.parseq_batch),
                     ("sar", S.doctr_batch)):
        check(f"{name}: no NaN/Inf", bool(np.isfinite(fn(crops)).all()))


def test_decode_charset():
    """index -> char agrees with the metadata shipped beside each model."""
    import numpy as np
    import ocr_supporting as S

    for name in S.MODEL_NAMES:
        _, meta_path = S.model_paths(name)
        if not os.path.isfile(meta_path):
            check(f"{name}: metadata present", False, meta_path)
            continue
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))

        if name == "catmus":
            idx2ch = {int(k): v for k, v in meta["idx2ch"].items()}
            check_eq("catmus: blank index is 0", int(meta["blank_index"]), 0)
            check("catmus: blank is not a character", 0 not in idx2ch)
            check_eq("catmus: classes = codec + blank",
                     int(meta["n_classes"]), len(idx2ch) + 1)
            # A hand-built CTC sequence must decode to a known string.
            n_classes = int(meta["n_classes"])
            want = "AB"
            a = next(i for i, c in idx2ch.items() if c == "A")
            b = next(i for i, c in idx2ch.items() if c == "B")
            seq = [0, a, a, 0, b, 0]          # blank, A repeated, blank, B, blank
            logits = np.full((1, len(seq), n_classes), -10.0, dtype=np.float32)
            for t, idx in enumerate(seq):
                logits[0, t, idx] = 10.0
            text, conf = S.decode_ctc(logits, idx2ch, blank=0)[0]
            check_eq("catmus: CTC collapses repeats and drops blanks", text, want)
            check("catmus: confidence in [0,1]", 0.0 <= conf <= 1.0, f"{conf:.4f}")

        elif name == "parseq":
            charset = meta["charset"]
            check_eq("parseq: charset length", len(charset), 94)
            check_eq("parseq: eos id", int(meta["eos_id"]), 0)
            # itos = ('[E]',) + charset, so id i>0 maps to charset[i-1].
            n = len(charset) + 1
            want = "A7"
            ids = [charset.index("A") + 1, charset.index("7") + 1, 0, 5]
            logits = np.full((1, len(ids), n), -10.0, dtype=np.float32)
            for t, idx in enumerate(ids):
                logits[0, t, idx] = 10.0
            text, _conf = S.decode_parseq(logits, charset)[0]
            check_eq("parseq: decodes and truncates at eos", text, want)

        else:
            vocab = meta["vocab"]
            specials = ["<eos>"] if name == "sar" else ["<eos>", "<sos>", "<pad>"]
            n = len(vocab) + len(specials)
            want = "A7"
            ids = [vocab.index("A"), vocab.index("7"), len(vocab), 3]
            logits = np.full((1, len(ids), n), -10.0, dtype=np.float32)
            for t, idx in enumerate(ids):
                logits[0, t, idx] = 10.0
            text, _ = S.decode_attention(logits, vocab, specials)[0]
            check_eq(f"{name}: decodes and stops at <eos>", text, want)
            check(f"{name}: vocab has no space (alignment must not assume one)",
                  " " not in vocab)


def test_batch_equals_single():
    """Decoding a batch equals decoding each crop alone.

    This is the bug class that bit PARSeq during conversion: a traced squeeze() that
    was correct at one batch size and wrong at another. It is invisible at N=1.
    """
    import numpy as np
    import ocr_supporting as S

    crops = [img for _, img in _load_crops(limit=6)]
    if not check("fixture crops present", bool(crops)):
        return

    for name in S.MODEL_NAMES:
        rec = S.get_recogniser(name)
        if rec is None:
            check(f"{name}: loaded", False, "model unavailable")
            continue
        batched = rec.recognise(crops)
        singles = [rec.recognise([c])[0] for c in crops]

        texts_ok = [b[0] for b in batched] == [s[0] for s in singles]
        check(f"{name}: batch text == single text", texts_ok,
              "" if texts_ok else f"batch={[b[0] for b in batched]} single={[s[0] for s in singles]}")
        confs_ok = all(abs(b[1] - s[1]) < 1e-3 for b, s in zip(batched, singles))
        check(f"{name}: batch confidence == single confidence", confs_ok)
        check_eq(f"{name}: one result per crop", len(batched), len(crops))

    # CATMuS batches crops of different widths into one padded tensor; the padding must
    # not change what the widest or the narrowest crop reads.
    rec = S.get_recogniser("catmus")
    if rec is not None:
        mixed = sorted(crops, key=lambda c: c.shape[1] / max(1, c.shape[0]))
        pair = [mixed[0], mixed[-1]]
        together = rec.recognise(pair)
        apart = [rec.recognise([c])[0] for c in pair]
        check("catmus: width padding does not change the reading",
              [t[0] for t in together] == [a[0] for a in apart],
              f"{[t[0] for t in together]} vs {[a[0] for a in apart]}")


def test_known_crop_roundtrip():
    """ONNX adapter reads what the reference implementation reads, on real crops."""
    import ocr_supporting as S

    if not check("reference readings present", REFERENCE.is_file(),
                 "run RESOURCES/supporting_reference_probe.py"):
        return
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    items = _load_crops()
    if not check("fixture crops present", bool(items)):
        return

    for name in S.MODEL_NAMES:
        ref = reference.get(name)
        if not ref:
            check(f"{name}: reference recorded", False)
            continue
        rec = S.get_recogniser(name)
        if rec is None:
            check(f"{name}: loaded", False, "model unavailable")
            continue
        got = rec.recognise([img for _, img in items])
        mismatches = []
        for (m, _), (text, _conf) in zip(items, got):
            want = ref.get(m["file"])
            if want is not None and text != want:
                mismatches.append((m["file"], want, text))
        check(f"{name}: matches the reference implementation on {len(items)} crops",
              not mismatches,
              "" if not mismatches else "; ".join(
                  f"{f}: ref={w!r} onnx={g!r}" for f, w, g in mismatches[:4]))

    # Ground truth, where a human confirmed it by looking at the crop. These models
    # disagree with each other by design, so this only asserts that at least one of
    # them reads each verified crop correctly -- ignoring space, which three of the
    # four cannot emit at all.
    truth = [(m, img) for m, img in items if m.get("expected")]
    if truth:
        readers = {n: S.get_recogniser(n) for n in S.MODEL_NAMES}
        readers = {n: r for n, r in readers.items() if r is not None}
        results = {n: r.recognise([img for _, img in truth]) for n, r in readers.items()}
        unread = []
        for i, (m, _) in enumerate(truth):
            want = m["expected"].replace(" ", "")
            if not any(results[n][i][0].replace(" ", "") == want for n in results):
                unread.append((m["file"], m["expected"],
                               {n: results[n][i][0] for n in results}))
        check(f"every one of {len(truth)} verified crops is read by at least one model",
              not unread,
              "" if not unread else "; ".join(f"{f}: want {w!r} got {g}" for f, w, g in unread[:3]))


def test_batch_is_capped_at_the_trt_profile():
    """A caller may pass any number of crops; the session must never see more than 32.

    MEASURED consequence of getting this wrong: handing PARSeq 489 crops produced
    "Set dimension [489,3,32,128] ... does not satisfy any optimization profiles", and
    _SafeSession then rebuilt the session WITHOUT TensorRT -- permanently, for the rest of
    the process. One oversized call silently drops the whole document onto the slow CUDA
    EP. A single page can produce hundreds of crops, so this is a normal case, not an edge.
    """
    import numpy as np
    import ocr_supporting as S

    check_eq("MAX_BATCH matches the TensorRT profile maximum", S.MAX_BATCH, 32)
    for name, profile in S._TRT_PROFILES.items():
        cap = int(profile["max_shape"].split("x")[0])
        check_eq(f"{name}: chunking cap equals its OWN profile maximum",
                 S.max_batch_for(name), cap)
        check(f"{name}: never chunks above the global cap", cap <= S.MAX_BATCH)
    # CATMuS is deliberately narrower: it cannot batch, and 32 cost 2.1GB of VRAM.
    check_eq("catmus is capped at 4", S.max_batch_for("catmus"), 4)

    crops = [img for _, img in _load_crops()]
    if not check("fixture crops present", bool(crops)):
        return
    many = (crops * 8)[:100]          # comfortably over the cap

    for name in S.MODEL_NAMES:
        rec = S.get_recogniser(name)
        if rec is None:
            check(f"{name}: loaded", False, "model unavailable")
            continue

        seen = []
        real_run = rec.sess.run

        def spy(output_names, input_feed, run_options=None, _r=real_run, _s=seen):
            for value in input_feed.values():
                _s.append(int(getattr(value, "shape", [0])[0]))
            return _r(output_names, input_feed, run_options)

        rec.sess.run = spy
        try:
            results = rec.recognise(many)
        finally:
            rec.sess.run = real_run

        check_eq(f"{name}: one result per crop", len(results), len(many))
        cap = S.max_batch_for(name)
        check(f"{name}: no call exceeded its profile cap of {cap} "
              f"(largest {max(seen) if seen else 0})",
              bool(seen) and max(seen) <= cap)
        check(f"{name}: it really did split into several calls", len(seen) > 1,
              f"{len(seen)} call(s)")

        # Chunking must not change a reading: the first crops read the same either way.
        small = rec.recognise(crops[:4])
        check(f"{name}: chunked results match an unchunked batch",
              [t for t, _ in results[:4]] == [t for t, _ in small],
              f"{[t for t, _ in results[:4]]} vs {[t for t, _ in small]}")

        # The session must still be on TensorRT -- i.e. nothing forced a rebuild.
        bound = rec.sess.get_providers()
        check(f"{name}: session was not downgraded off TensorRT",
              "TensorrtExecutionProvider" in bound, str(bound))


def test_missing_model_degrades():
    """A model whose file is absent is skipped with a warning, never fatal."""
    import ocr_supporting as S

    saved_dir = os.environ.get("OCULI_SUPPORTING_MODEL_DIR")
    saved_cache = getattr(S._TLS, "cache", None)
    try:
        S._TLS.cache = {}
        os.environ["OCULI_SUPPORTING_MODEL_DIR"] = os.path.join(
            os.environ.get("TEMP", "."), "oculi-no-such-model-dir")

        check_eq("nothing reports as available", S.available(),
                 {n: False for n in S.MODEL_NAMES})
        for name in S.MODEL_NAMES:
            check(f"{name}: returns None rather than raising",
                  S.get_recogniser(name) is None)
        check_eq("prewarm reports every model unloaded", S.prewarm(),
                 {n: False for n in S.MODEL_NAMES})
    finally:
        if saved_dir is None:
            os.environ.pop("OCULI_SUPPORTING_MODEL_DIR", None)
        else:
            os.environ["OCULI_SUPPORTING_MODEL_DIR"] = saved_dir
        S._TLS.cache = saved_cache if saved_cache is not None else {}

    # An unknown name is a programming error, not a degradation.
    try:
        S.get_recogniser("nope")
        check("unknown model name raises", False)
    except KeyError:
        check("unknown model name raises", True)

    # ...and the real directory still works afterwards.
    check("real model dir still resolves", any(S.available().values()),
          str(S.available()))


if __name__ == "__main__":
    sys.exit(main(sys.modules[__name__]))
