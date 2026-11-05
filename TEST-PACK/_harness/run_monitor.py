"""Passive run monitor - the window into the machine during a long extraction.

DIAGNOSTIC ONLY. Starts nothing, changes nothing, imports nothing from bk-main.
Run it in its own console BEFORE starting a long run, and leave it going.

Why this exists: the failures under investigation are ACCUMULATED - they appear only
after a day or more of running, which is exactly when nobody is watching. A single
end-of-run traceback cannot distinguish "gradually leaked until it burst" from "one
page happened to be too big". A time series can.

It samples every OCULI-related process (the desktop app, session_runtime, page workers
and any paddle_ipc subprocess) and writes one CSV row per process per sample:

    ts, pid, name, tag, rss_mb, vms_mb, threads, handles, cpu_pct, sys_avail_mb

and flags, in a separate EVENTS log, the moments that matter:

  * a paddle_ipc subprocess appearing or DISAPPEARING - the transport errors in the
    reported log are preceded by that server dying, and its death is what makes the
    in-process paddle fallback reachable
  * total RSS crossing a threshold, alongside system available memory
  * any process whose RSS rises on every sample across a window (leak signature)
  * thread counts, because a thread-local PaddleOCR engine is ~1 GB, so thread growth
    inside a page worker predicts memory growth

Usage:
    python run_monitor.py
    python run_monitor.py --interval 30 --out D:\\diag
    python run_monitor.py --total-warn-gb 24

Needs psutil:  python -m pip install psutil
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime

try:
    import psutil
except ImportError:
    print("psutil is required:  python -m pip install psutil", file=sys.stderr)
    raise SystemExit(2)

# Processes worth watching, matched on exe name and then confirmed on the command line.
NAME_HINTS = ("documentextractionconsole", "python", "pythonw")
CMD_HINTS = ("session_runtime", "pipeline", "paddle_ipc", "bk-main", "oculi")


def _tag(proc):
    """Short label for what this process IS, so rows read without the full cmdline."""
    try:
        cmd = " ".join(proc.cmdline()).lower()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return "?"
    if "paddle_ipc" in cmd and "--server-shared" in cmd:
        return "paddle-ipc-shared"
    if "paddle_ipc" in cmd:
        return "paddle-ipc"
    for key in ("session_runtime", "pipeline", "documentextractionconsole"):
        if key in cmd:
            return key
    return "other"


_SELF_PID = os.getpid()


def _interesting(proc):
    # Exclude the monitor itself: it lives under c:\OCULI.pd1\, so "oculi" matches its own
    # command line and it would otherwise report itself as a watched process.
    if proc.pid == _SELF_PID:
        return False
    try:
        if not proc.name().lower().startswith(NAME_HINTS):
            return False
        cmd = " ".join(proc.cmdline()).lower()
        if "run_monitor.py" in cmd:
            return False
        return any(h in cmd for h in CMD_HINTS)
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=15.0, help="seconds between samples")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--total-warn-gb", type=float, default=0.0,
                    help="EVENT when total OCULI RSS crosses this (0 = auto: 70 pct of RAM)")
    ap.add_argument("--growth-samples", type=int, default=8,
                    help="consecutive rising samples that count as a leak signature")
    args = ap.parse_args()

    total_ram_gb = psutil.virtual_memory().total / 1024 ** 3
    warn_gb = args.total_warn_gb or (total_ram_gb * 0.70)

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.out, "oculi_monitor_" + stamp + ".csv")
    evt_path = os.path.join(args.out, "oculi_events_" + stamp + ".log")

    print("machine RAM       : %.1f GB" % total_ram_gb)
    print("total RSS warning : %.1f GB" % warn_gb)
    print("samples -> " + csv_path)
    print("events  -> " + evt_path)
    print("Ctrl+C to stop.\n")

    hist = defaultdict(lambda: deque(maxlen=args.growth_samples))
    seen_pids = {}
    warned_growth = set()
    warned_total = False

    evt = open(evt_path, "a", encoding="utf-8", buffering=1)

    def event(msg):
        line = datetime.now().isoformat(timespec="seconds") + "  " + msg
        print(line, flush=True)
        evt.write(line + "\n")

    event("monitor started (interval %.0fs, RAM %.1f GB)" % (args.interval, total_ram_gb))

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "pid", "name", "tag", "rss_mb", "vms_mb",
                    "threads", "handles", "cpu_pct", "sys_avail_mb"])
        try:
            while True:
                now = datetime.now().isoformat(timespec="seconds")
                avail_mb = psutil.virtual_memory().available / 1024 ** 2
                total_rss = 0.0
                alive = {}

                for proc in psutil.process_iter(["pid", "name"]):
                    if not _interesting(proc):
                        continue
                    try:
                        with proc.oneshot():
                            mem = proc.memory_info()
                            rss_mb = mem.rss / 1024 ** 2
                            tag = _tag(proc)
                            alive[proc.pid] = tag
                            total_rss += rss_mb
                            try:
                                handles = proc.num_handles()
                            except (AttributeError, psutil.AccessDenied):
                                handles = -1
                            w.writerow([now, proc.pid, proc.name(), tag,
                                        "%.1f" % rss_mb, "%.1f" % (mem.vms / 1024 ** 2),
                                        proc.num_threads(), handles,
                                        "%.1f" % proc.cpu_percent(None), "%.0f" % avail_mb])

                            # Leak signature: RSS strictly rising for the whole window.
                            h = hist[proc.pid]
                            h.append(rss_mb)
                            if (len(h) == h.maxlen
                                    and proc.pid not in warned_growth
                                    and all(b > a for a, b in zip(h, list(h)[1:]))):
                                warned_growth.add(proc.pid)
                                event("LEAK-SIGNATURE pid=%d (%s) RSS rose every sample for "
                                      "%d samples: %.0f -> %.0f MB"
                                      % (proc.pid, tag, h.maxlen, h[0], h[-1]))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                fh.flush()

                # paddle_ipc lifecycle - its death is what triggers the in-process fallback.
                for pid, tag in alive.items():
                    if pid not in seen_pids:
                        event("process APPEARED pid=%d (%s)" % (pid, tag))
                for pid, tag in seen_pids.items():
                    if pid not in alive:
                        if tag.startswith("paddle-ipc"):
                            event("!! process EXITED  pid=%d (%s)  <-- in-process paddle "
                                  "fallback becomes reachable from here" % (pid, tag))
                        else:
                            event("process EXITED   pid=%d (%s)" % (pid, tag))
                seen_pids = alive

                gb = total_rss / 1024
                if gb > warn_gb and not warned_total:
                    warned_total = True
                    event("TOTAL RSS %.1f GB crossed the %.1f GB threshold "
                          "(system available %.1f GB)" % (gb, warn_gb, avail_mb / 1024))
                elif gb < warn_gb * 0.85:
                    warned_total = False

                time.sleep(args.interval)
        except KeyboardInterrupt:
            event("monitor stopped")
        finally:
            evt.close()

    print("\nWrote " + csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
