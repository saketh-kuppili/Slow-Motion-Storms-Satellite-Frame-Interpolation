"""The ``sms`` command line: one command per pipeline step (run ``sms --help``)."""

from __future__ import annotations

import argparse
import json
import sys

from . import acquire
from .config import load_config

STEPS = """
Phase 1  spec                  check the experimental specification
Phase 2  download              download full-cadence Band 13 (+ sampled NOAA CMIP references)
         (event selection, run once: python scripts/select_events.py tracks | verify | rank-days)
         manifest-verify       recompute the SHA-256 of every downloaded file
Phase 3  process               calibrate, quality-check, segment and store
         figures               documentation figures from real data
Phase 4  build-dataset         windows, patches, splits, normalisation statistics
         validate-dataset      independent validation report
         summarize             final dataset summary
"""


def cmd_spec(args) -> int:
    from .config import resolve_spec, write_spec

    spec = resolve_spec(load_config())
    print(f"specification consistent; written to {write_spec(spec)}")
    for g in spec["windows"]["gaps"]:
        print(f"  gap {g['gap_min']} min: {g['core_frames']} core frames, {g['hidden_frames']} hidden")
    print(f"  splits: {spec['splits']}")
    return 0


def cmd_download(args) -> int:
    cfg = load_config()
    offset, duration, events = args.offset_min, args.duration_min, args.events
    if args.pilot:
        offset, duration, events = cfg["pilot"]["offset_min"], cfg["pilot"]["duration_min"], cfg["pilot"]["events"]
    if not events:
        sys.exit("name the events to download (--events) or use --pilot")
    for e in events:
        rows = acquire.download_event(e, cfg, offset, duration, with_references=not args.no_references)
        print(f"{e}: {sum(r['purpose'] == 'dataset' for r in rows)} L1b files, "
              f"{sum(r['purpose'] == 'calibration_reference' for r in rows)} CMIP references, "
              f"{sum(r['size_bytes'] for r in rows) / 1e6:.1f} MB")
    return 0


def cmd_manifest_verify(args) -> int:
    rows = acquire.read_manifest()
    problems = acquire.verify_manifest(rows)
    print(f"{len(rows)} manifest rows, {len(problems)} problems", *problems[:20], sep="\n")
    return 1 if problems else 0


def cmd_process(args) -> int:
    from .preprocess import process_event

    cfg = load_config()
    for e in args.events or sorted({r["event_id"] for r in acquire.read_manifest() if r["purpose"] == "dataset"}):
        s = process_event(e, cfg)
        worst = max((c["max_abs_diff_K"] for c in s["calibration_checks"]), default=float("nan"))
        print(f"{e}: {s['n_files']} scans -> {s['n_segments']} segment(s); {len(s['calibration_checks'])} calibration "
              f"checks, worst |dT| {worst:.4f} K, passed={s['calibration_passed']}; {len(s['scan_qc_problems'])} scans with QC notes")
    return 0


def cmd_figures(args) -> int:
    from .plots import make_figures

    print(f"{len(make_figures(load_config()))} figures written to reports/figures/")
    return 0


def cmd_build(args) -> int:
    from .dataset import build_dataset

    meta = build_dataset(load_config())
    print(json.dumps({"n_windows": meta["n_windows"], "n_patches": meta["n_patches"], "splits": meta["splits"]}, indent=2))
    return 0


def cmd_validate(args) -> int:
    from .validate import validate_dataset

    result = validate_dataset(load_config())
    for c in result["checks"]:
        print(f"  [{'pass' if c['passed'] else 'FAIL'}] {c['name']}: {c['n_checked']} checked, {c['n_failures']} failures")
    print("overall:", "PASS" if result["passed"] else "FAIL")
    return 0 if result["passed"] else 1


def cmd_summarize(args) -> int:
    from .validate import build_summary

    t = build_summary(load_config())["totals"]
    print(f"{t['n_events']} events, {t['n_scans']} scans, {t['n_windows']} windows, {t['n_patches']} patches; "
          "written to reports/dataset/final_summary.md")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sms", description=STEPS, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("spec").set_defaults(func=cmd_spec)
    dl = sub.add_parser("download")
    dl.add_argument("--events", nargs="+")
    dl.add_argument("--pilot", action="store_true", help="the pilot slice from configs/data/base.yaml")
    dl.add_argument("--offset-min", type=float)
    dl.add_argument("--duration-min", type=float)
    dl.add_argument("--no-references", action="store_true")
    dl.set_defaults(func=cmd_download)
    sub.add_parser("manifest-verify").set_defaults(func=cmd_manifest_verify)
    pr = sub.add_parser("process")
    pr.add_argument("--events", nargs="+")
    pr.set_defaults(func=cmd_process)
    for name, func in (("figures", cmd_figures), ("build-dataset", cmd_build), ("validate-dataset", cmd_validate),
                       ("summarize", cmd_summarize)):
        sub.add_parser(name).set_defaults(func=func)
    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
