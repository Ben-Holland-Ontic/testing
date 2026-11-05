"""Identical classification battery, run against any local ONNX LLM.

Takes the model directory as argv[1] so Phi-4-mini and Phi-4 14B are compared on exactly
the same prompts, renderings and cases -- the one variable being model size.

Renderings come from canonical_view. Cases are region groups from a captured
DataCleaningInputsV2, labelled by whether the pipeline's own grid detector found them to be
part of a grid. That label is used ONLY for scoring, never shown to the model.

    $PY = 'C:\OCULI.pd1\TESTING\.venv-genai122\Scripts\python.exe'
    & $PY TESTING\RESOURCES\llm_region_battery.py <model_dir> [capture.json]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phi_smoke_probe import register_cuda_dll_dirs  # noqa: E402
import canonical_view as CV  # noqa: E402

register_cuda_dll_dirs()
import onnxruntime_genai as og  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else None
CAPTURE = sys.argv[2] if len(sys.argv) > 2 else r"C:\OCULI.pd1\TESTING\RESULTS\cleaning_inputs_full.json"


def main() -> int:
    cfg = Path(MODEL) / "genai_config.json"
    c = json.loads(cfg.read_text(encoding="utf-8"))
    so = c["model"]["decoder"]["session_options"]
    if not so.get("provider_options"):
        so["provider_options"] = [{"cuda": {}}]
        cfg.write_text(json.dumps(c, indent=2), encoding="utf-8")
        print("patched genai_config.json -> cuda")

    t0 = time.perf_counter()
    model = og.Model(MODEL)
    tok = og.Tokenizer(model)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s  ({Path(MODEL).parts[-3]})")

    def ask(payload: str) -> str:
        q = ("<|system|>You judge a group of detected regions from an engineering drawing. "
             "Answer with ONE word: TABLE or INFO_CLUSTER. Nothing else.<|end|><|user|>"
             f"ASCII spatial map of all regions in one area of the drawing:\n\n{payload}\n\n"
             "Do these regions together form a TABLE, or are they an INFO_CLUSTER of "
             "unrelated fields? One word.<|end|><|assistant|>")
        t = tok.encode(q)
        pr = og.GeneratorParams(model)
        pr.set_search_options(max_length=len(t) + 8, do_sample=False)
        g = og.Generator(model, pr)
        g.append_tokens(t)
        s = tok.create_stream()
        out = []
        while not g.is_done():
            g.generate_next_token()
            try:
                out.append(s.decode(g.get_next_tokens()[0]))
            except Exception:
                pass
        a = "".join(out).upper()
        if "TABLE" in a and "INFO" not in a:
            return "TABLE"
        return "INFO_CLUSTER" if "INFO" in a else "?"

    pages = json.loads(Path(CAPTURE).read_text(encoding="utf-8"))
    cases = []
    p1 = [x for x in pages if x["page_id"].endswith("PG001")][0]
    cases.append(("PG001 the 7x3 table", "TABLE", CV._cells(p1, 2500, 560)))
    for p in pages:
        cf = p["cluster_features"]
        pg = p["page_id"][-5:]
        g = [c for c in CV._cells(p) if (cf.get(c["poly_id"]) or {}).get("is_part_of_grid")]
        n = [c for c in CV._cells(p) if not (cf.get(c["poly_id"]) or {}).get("is_part_of_grid")]
        if len(g) >= 4:
            cases.append((f"{pg} grid({len(g)})", "TABLE", g))
        if len(n) >= 4:
            cases.append((f"{pg} nongrid({len(n)})", "INFO_CLUSTER", n))

    ok = 0
    t0 = time.perf_counter()
    print(f"\n{'case':<23} {'expected':<14} {'got':<14}")
    for name, exp, cells in cases:
        got = ask(CV.render_map(cells, 90, 30))
        ok += got == exp
        print(f"{name:<23} {exp:<14} {got:<14} {'OK' if got == exp else 'MISS'}")
    el = time.perf_counter() - t0
    print(f"\nSCORE {ok}/{len(cases)}   ({el:.1f}s, {el / len(cases) * 1000:.0f} ms/case)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
