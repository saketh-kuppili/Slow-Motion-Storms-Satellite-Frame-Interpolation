"""Phase 4: independent validation of the built dataset, and the final dataset summary."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .acquire import read_csv, read_event_windows, read_manifest
from .config import REPO_ROOT, SPLITS, frames_per_gap, from_portable, load_events, load_splits, manifest_dir, \
    reports_dir, split_of
from .dataset import (core_indices, dataset_dir, denormalize, load_index, new_moments, normalize,
                      shared_source_issues, std_of, temporal_separation_issues, update_moments)
from .preprocess import (brightness_temperature, open_segment, read_frames, read_scan, read_segment,
                         read_segment_index, valid_mask)

# Every GOES-East ABI fixed-grid product carries these (PUG Vol. 3, Table 5.1.2.8).
EXPECTED_PROJECTION = {"perspective_point_height": 35786023.0, "semi_major_axis": 6378137.0,
                       "semi_minor_axis": 6356752.31414, "longitude_of_projection_origin": -75.0, "sweep_angle_axis": "x"}
ABI_2KM_STEP_RAD = 5.6e-5
CHECKS = ("segment_projection_and_grid", "raw_file_grid_reconfirmed", "core_timing", "t_fractions_recomputed",
          "context_frames", "context_absence_genuine", "patch_bounds_and_validity", "bt_physical_range",
          "split_consistency", "leakage", "source_content_disjoint", "train_only_statistics",
          "normalization_round_trip", "calibration_reports")


def validate_dataset(cfg: dict) -> dict:
    """Recompute everything the index claims from the segment files and original downloads; report each check."""
    windows, patches = load_index(allow_test=True)
    meta = json.loads((dataset_dir() / "dataset_meta.json").read_text(encoding="utf-8"))
    win, cadence, usable = cfg["windows"], cfg["instrument"]["nominal_cadence_s"], cfg["quality"]["usable_dqf"]
    lo_phys, hi_phys = cfg["calibration"]["physical_range_K"]
    n = dict.fromkeys(CHECKS, 0)  # items checked
    fail = {c: [] for c in CHECKS}  # failure messages
    manifest = {r["key"]: r for r in read_manifest() if r["purpose"] == "dataset"}

    # Segments: projection metadata, regular 56 urad grid, and three frames re-derived from the original files.
    segs = {}
    for sp in sorted({w["segment_path"] for w in windows}):
        seg = read_segment(from_portable(sp))
        segs[sp], sid = seg, seg["attrs"]["segment_id"]
        n["segment_projection_and_grid"] += 1
        for key, want in EXPECTED_PROJECTION.items():
            got = seg["proj"].get(key)
            if not (got == want if isinstance(want, str) else got is not None and abs(float(got) - want) <= 1e-6):
                fail["segment_projection_and_grid"].append(f"{sid}: projection {key}={got!r}")
        for name in ("x", "y"):
            if not np.allclose(np.abs(np.diff(seg[name])), ABI_2KM_STEP_RAD, rtol=0, atol=1e-9):
                fail["segment_projection_and_grid"].append(f"{sid}: {name} spacing not {ABI_2KM_STEP_RAD} rad")
        if [seg["y"].size, seg["x"].size] != list(cfg["quality"]["expected_shape_px"]):
            fail["segment_projection_and_grid"].append(f"{sid}: shape")
        nf = len(seg["time_mid"])
        for k in sorted({0, nf // 2, nf - 1}):
            row = manifest.get(seg["source_keys"][k])
            if row is None or not from_portable(row["local_path"]).exists():
                continue
            n["raw_file_grid_reconfirmed"] += 1
            scan, msg = read_scan(from_portable(row["local_path"])), f"{sid} frame {k}"
            if scan["grid"] != seg["attrs"]["grid_digest"]:
                fail["raw_file_grid_reconfirmed"].append(f"{msg}: raw grid digest differs")
            if not (np.array_equal(scan["x"], seg["x"]) and np.array_equal(scan["y"], seg["y"])):
                fail["raw_file_grid_reconfirmed"].append(f"{msg}: raw x/y differ from stored coordinates")
            if any(seg["proj"].get(key) != v for key, v in scan["proj"].items()):
                fail["raw_file_grid_reconfirmed"].append(f"{msg}: projection differs")
            stored = read_frames(seg["path"], k)[0]
            recomputed = brightness_temperature(scan["radiance"], scan["planck"]).astype(np.float32)
            if not np.array_equal(stored, recomputed, equal_nan=True):
                fail["raw_file_grid_reconfirmed"].append(f"{msg}: stored BT differs from recomputed BT")

    # Windows: timing, normalised times, context frames present and genuinely absent.
    for w in windows:
        t, gap, idx, wid = segs[w["segment_path"]]["time_mid"], int(w["gap_min"]), core_indices(w), w["window_id"]
        steps_n = frames_per_gap(gap, cadence)
        n["core_timing"] += 1
        if len(idx) != steps_n + 1 or int(w["n_hidden"]) != steps_n - 1 or np.any(np.diff(idx) <= 0) \
                or (idx[0], idx[-1]) != (int(w["core_first_index"]), int(w["core_last_index"])):
            fail["core_timing"].append(f"{wid}: core indices inconsistent")
            continue
        core = t[idx]
        span = core[-1] - core[0]
        if np.any(np.abs(np.diff(core) - cadence) > win["cadence_tolerance_s"]) or \
                abs(span - gap * 60) > win["endpoint_gap_tolerance_s"]:
            fail["core_timing"].append(f"{wid}: core interval or endpoint gap outside tolerance")
        n["t_fractions_recomputed"] += 1
        stored_t, real_t = np.array([float(x) for x in w["t_fractions"].split(";")]), (core - core[0]) / span
        if np.max(np.abs(stored_t - real_t)) > 1e-5 or stored_t[0] != 0 or stored_t[-1] != 1 or \
                np.max(np.abs(real_t - np.arange(steps_n + 1) / steps_n)) > win["hidden_time_tolerance"]:
            fail["t_fractions_recomputed"].append(f"{wid}: normalised times")
        for name, off in zip(("before", "after"), win["context_offset_gaps"]):
            target = core[0] + off * gap * 60
            allowed = np.arange(0, idx[0]) if name == "before" else np.arange(idx[-1] + 1, len(t))
            if w[f"ctx_{name}_available"] == "1":
                n["context_frames"] += 1
                j = int(w[f"ctx_{name}_index"])
                if j not in allowed or abs(t[j] - target) > win["context_tolerance_s"] or \
                        abs((t[j] - core[0]) / span - float(w[f"ctx_{name}_offset_gap_units"])) > 1e-5:
                    fail["context_frames"].append(f"{wid}: context {name}")
            else:
                n["context_absence_genuine"] += 1
                if allowed.size and np.min(np.abs(t[allowed] - target)) <= win["context_tolerance_s"]:
                    fail["context_absence_genuine"].append(f"{wid}: context {name} marked missing but a frame exists")

    # Patches: bounds, validity in every core frame, physical brightness temperatures.
    by_window = {w["window_id"]: w for w in windows}
    H, W = cfg["quality"]["expected_shape_px"]
    open_files = {sp: open_segment(from_portable(sp)) for sp in segs}
    for p in patches:
        n["patch_bounds_and_validity"] += 1
        w = by_window.get(p["window_id"])
        r0, c0, size = int(p["row0"]), int(p["col0"]), int(p["size"])
        if w is None or size != cfg["patches"]["size_px"] or r0 < 0 or c0 < 0 or r0 + size > H or c0 + size > W:
            fail["patch_bounds_and_validity"].append(f"{p['patch_id']}: unknown window or out of bounds")
            continue
        if (p["split"], p["event_id"]) != (w["split"], w["event_id"]):
            fail["split_consistency"].append(f"{p['patch_id']}: split/event differs from its window")
        bt, dqf = read_frames(open_files[w["segment_path"]], core_indices(w), slice(r0, r0 + size), slice(c0, c0 + size))
        valid = valid_mask(bt, dqf, usable)
        frac = float(valid.all(axis=0).mean())
        if frac + 1e-12 < cfg["patches"]["min_core_valid_fraction"] or abs(frac - float(p["core_valid_fraction"])) > 1e-5:
            fail["patch_bounds_and_validity"].append(f"{p['patch_id']}: core valid fraction {frac:.6f}")
        n["bt_physical_range"] += 1
        vals = bt[valid]
        if vals.size and (vals.min() < lo_phys or vals.max() > hi_phys):
            fail["bt_physical_range"].append(f"{p['patch_id']}: BT {vals.min():.2f}-{vals.max():.2f} K")

    # Splits and leakage: assignment, temporal separation, no shared file names or file contents.
    owner = split_of(load_splits(REPO_ROOT / cfg["splits"]["file"], load_events()["events"]))
    sources = defaultdict(set)
    for w in windows:
        n["split_consistency"] += 1
        if owner.get(w["event_id"]) != w["split"]:
            fail["split_consistency"].append(f"{w['window_id']}: split {w['split']} but event in {owner.get(w['event_id'])}")
        used = core_indices(w) + [int(w[f"ctx_{s}_index"]) for s in ("before", "after") if w[f"ctx_{s}_available"] == "1"]
        sources[w["split"]].update(segs[w["segment_path"]]["source_keys"][k] for k in used)
    n["leakage"] = len(windows)
    intervals = {e: (v["start"], v["end"]) for e, v in read_event_windows().items() if e in owner}
    fail["leakage"] += temporal_separation_issues(owner, intervals, cfg["splits"]["min_event_separation_hours"])
    fail["leakage"] += shared_source_issues(sources)
    digests = {s: set() for s in sources}
    for s, keys in sources.items():
        for k in keys:
            n["source_content_disjoint"] += 1
            if k in manifest:
                digests[s].add(manifest[k]["sha256"])
            else:
                fail["source_content_disjoint"].append(f"{k}: not in the manifest")
    fail["source_content_disjoint"] += shared_source_issues(digests)

    # Normalisation statistics: training events only, and reproduced by an independent pass.
    z, train_events = meta.get("zscore_reference_train_only"), {e for e, s in owner.items() if s == "train"}
    if not z:
        fail["train_only_statistics"].append("no training statistics in dataset_meta.json")
    else:
        if set(z["fitted_on"]) - train_events or set(z["fitted_on"]) != {w["event_id"] for w in windows if w["split"] == "train"}:
            fail["train_only_statistics"].append(f"fitted_on {sorted(z['fitted_on'])} is not exactly the training events")
        m, seen = new_moments(), set()
        for w in windows:
            if w["split"] == "train":
                for k in core_indices(w):
                    if (w["segment_path"], k) not in seen:
                        seen.add((w["segment_path"], k))
                        bt, dqf = read_frames(open_files[w["segment_path"]], k)
                        update_moments(m, bt[valid_mask(bt, dqf, usable)])
        n["train_only_statistics"] = m["n"]
        for name, ours in (("offset_K", m["mean"]), ("scale_K", std_of(m))):
            if not math.isclose(ours, z[name], rel_tol=1e-9, abs_tol=1e-9):
                fail["train_only_statistics"].append(f"{name}: recomputed {ours!r} != stored {z[name]!r}")

    for ds in open_files.values():
        ds.close()

    off, scale = meta["normalization"]["offset_K"], meta["normalization"]["scale_K"]
    probe = np.linspace(150.0, 345.0, 100001, dtype=np.float32)
    err = float(np.max(np.abs(denormalize(normalize(probe, off, scale), off, scale) - probe)))
    n["normalization_round_trip"] = probe.size
    if err > 1e-4:
        fail["normalization_round_trip"].append(f"float32 round-trip error {err:.2e} K")

    calib = []
    for e in sorted({w["event_id"] for w in windows}):
        path = reports_dir() / "preprocess" / f"{e}_C{cfg['instrument']['bands'][0]:02d}.json"
        checks = json.loads(path.read_text(encoding="utf-8")).get("calibration_checks", []) if path.exists() else []
        if not checks:
            fail["calibration_reports"].append(f"{e}: no calibration comparison")
        for c in checks:
            n["calibration_reports"] += 1
            calib.append(c)
            if not c["passed"]:
                fail["calibration_reports"].append(f"{e} {c['key']}: max |dT| {c['max_abs_diff_K']:.4f} K")

    def describe(col):
        a = np.array([float(w[col]) for w in windows if w[col] != ""])
        return None if not a.size else {"n": int(a.size), "min": float(a.min()), "mean": float(a.mean()),
                                         "max": float(a.max()), "std": float(a.std())}

    summary = {"n_windows": len(windows), "n_patches": len(patches),
               "context_before_available_fraction": float(np.mean([w["ctx_before_available"] == "1" for w in windows])),
               "context_after_available_fraction": float(np.mean([w["ctx_after_available"] == "1" for w in windows])),
               **{c: describe(c) for c in ("ctx_before_offset_s", "ctx_after_offset_s", "ctx_before_offset_gap_units",
                                           "ctx_after_offset_gap_units")},
               "endpoint_span_s": describe("span_s"), "max_t_deviation": describe("max_t_deviation"),
               "normalization_round_trip_max_error_K": err,
               "calibration_max_abs_diff_K": max((c["max_abs_diff_K"] for c in calib), default=None),
               "calibration_n_scans_compared": len(calib)}
    result = {"passed": not any(fail.values()),
              "checks": [{"name": c, "passed": not fail[c], "n_checked": n[c], "n_failures": len(fail[c]),
                          "first_failures": fail[c][:10]} for c in CHECKS],
              "summary": summary}
    out = reports_dir() / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"# Dataset validation\n\nOverall: **{'PASS' if result['passed'] else 'FAIL'}**\n",
             _table(["Check", "Result", "Items checked", "Failures"],
                    [[c["name"], "pass" if c["passed"] else "FAIL", c["n_checked"], c["n_failures"]] for c in result["checks"]]),
             "\n## Summary\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n"]
    lines += [f"### Failures: {c['name']}\n\n" + "\n".join(f"- {m}" for m in c["first_failures"]) + "\n"
              for c in result["checks"] if c["first_failures"]]
    (out / "validation.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def _table(header: list, rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def _quantiles(edges: np.ndarray, counts: np.ndarray, qs) -> dict:
    """Quantiles from the stored histogram, interpolating linearly inside a bin (0.5 K resolution)."""
    cdf = np.cumsum(counts) / counts.sum()
    out = {}
    for q in qs:
        k = int(np.searchsorted(cdf, q))
        lo = cdf[k - 1] if k > 0 else 0.0
        frac = (q - lo) / (cdf[k] - lo) if cdf[k] > lo else 0.0
        out[f"p{int(round(q * 100)):02d}_K"] = float(edges[k] + frac * (edges[k + 1] - edges[k]))
    return out


def build_summary(cfg: dict) -> dict:
    """Final dataset summary (events, splits, counts, context, BT distribution, missing data, storage, validation)."""
    events_cfg = load_events()["events"]
    owner = split_of(load_splits(REPO_ROOT / cfg["splits"]["file"]))
    meta = json.loads((dataset_dir() / "dataset_meta.json").read_text(encoding="utf-8"))
    vpath = reports_dir() / "dataset" / "validation.json"
    validation = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else None
    windows, manifest, segments = read_event_windows(), read_manifest(), read_segment_index()
    selection = {r["event_id"]: r for r in read_csv(manifest_dir() / "event_selection.csv")}
    band = cfg["instrument"]["bands"][0]

    events, missing = [], {}
    for e in sorted(owner):
        rep_path = reports_dir() / "preprocess" / f"{e}_C{band:02d}.json"
        rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
        segs, win, ev = [s for s in segments if s["event_id"] == e], windows.get(e), events_cfg[e]
        stats = rep.get("interval_statistics", {})
        events.append({
            "event_id": e, "name": ev["name"], "type": ev["type"], "split": owner[e], "satellite": ev["satellite"],
            "sector": selection.get(e, {}).get("sector", ""),
            "window_start_utc": win["start"].isoformat() if win else "", "window_end_utc": win["end"].isoformat() if win else "",
            "n_scans_downloaded": sum(r["event_id"] == e and r["purpose"] == "dataset" for r in manifest),
            "n_scans_processed": rep.get("n_files"), "n_segments": len(segs),
            "segment_breaks": [s["break_reason_before"] for s in segs],
            "median_interval_s": stats.get("median_s"), "max_interval_s": stats.get("max_s"),
            "n_holes": sum(int(s["n_holes"]) for s in segs), "valid_fraction": rep.get("valid_fraction"),
            "n_scans_with_qc_notes": len(rep.get("scan_qc_problems", [])),
            "n_duplicates_dropped": rep.get("n_duplicates_dropped"),
            "calibration_checks": len(rep.get("calibration_checks", [])),
            "calibration_max_abs_diff_K": max((c["max_abs_diff_K"] for c in rep.get("calibration_checks", [])), default=None),
            **meta.get("per_event", {}).get(e, {}),
            "window_start_outcomes": meta.get("window_start_outcomes", {}).get(e, {})})
        dqf = defaultdict(int)
        qc = read_csv(reports_dir() / "preprocess" / f"{e}_C{band:02d}_scan_qc.csv")
        for row in qc:
            for k, v in json.loads(row["dqf_counts"]).items():
                dqf[k] += int(v)
        total = sum(dqf.values())
        expected = int(round(cfg["coverage"]["window_hours"] * 3600 / (stats.get("median_s") or cfg["instrument"]["nominal_cadence_s"])))
        missing[e] = {"expected_scans_at_observed_cadence": expected, "scans": len(qc), "missing_scans": expected - len(qc),
                      "pixel_dqf_fractions": {k: v / total for k, v in sorted(dqf.items())} if total else {},
                      "invalid_pixel_fraction": (1 - dqf.get("good", 0) / total) if total else None}

    hist = meta["bt_histogram_core_frames"]
    edges = np.asarray(hist["bin_edges_K"])
    centres = (edges[:-1] + edges[1:]) / 2
    bt = {}
    for s, counts in hist["counts"].items():
        c = np.asarray(counts)
        if c.sum():
            bt[s] = {**meta["pixel_statistics_core_frames"][s], **_quantiles(edges, c, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)),
                     "fraction_below_220K": float(c[centres < 220].sum() / c.sum()),
                     "fraction_below_200K": float(c[centres < 200].sum() / c.sum())}
    outcomes = defaultdict(int)
    for ev in events:
        for k, v in ev["window_start_outcomes"].items():
            outcomes[k] += v
    raw = defaultdict(int)
    for r in manifest:
        raw[r["purpose"]] += r["size_bytes"]
    in_splits = [s for s in segments if s["event_id"] in owner]
    n_frames, seg_bytes = sum(int(s["n_frames"]) for s in in_splits), sum(int(s["size_bytes"]) for s in in_splits)
    H, W = cfg["quality"]["expected_shape_px"]
    core = 1 + max(cfg["windows"]["gaps_min"]) * 60 // int(cfg["instrument"]["nominal_cadence_s"])
    storage = {"raw_bytes_by_purpose": dict(raw), "processed_segments_bytes": seg_bytes, "processed_frames": n_frames,
               "bytes_per_processed_frame": seg_bytes / n_frames if n_frames else None,
               "uncompressed_bytes_per_frame": H * W * 5,  # float32 BT + uint8 DQF
               "dataset_index_bytes": sum(f.stat().st_size for f in Path(dataset_dir()).rglob("*") if f.is_file()),
               "materialised_patches_uncompressed_bytes": meta["n_patches"] * core * cfg["patches"]["size_px"] ** 2 * 4}
    splits = {}
    for s in SPLITS:
        evs = [e for e in events if e["split"] == s]
        splits[s] = {"n_events": len(evs), "events": [e["event_id"] for e in evs],
                     "satellites": sorted({e["satellite"] for e in evs}), "n_scans": sum(e["n_scans_processed"] or 0 for e in evs),
                     **{k: meta["splits"][s][k] for k in ("n_windows", "n_patches", "n_windows_without_patch",
                                                         "n_ctx_before", "n_ctx_after", "n_ctx_both")}}
    summary = {"dataset": meta["dataset"], "processing_version": meta["processing_version"], "created_utc": meta["created_utc"],
               "events": events, "splits": splits,
               "totals": {"n_events": len(events), "n_scans": sum(e["n_scans_processed"] or 0 for e in events),
                          "n_windows": meta["n_windows"], "n_patches": meta["n_patches"]},
               "bt_distribution": bt, "missing_data": missing, "window_start_outcomes": dict(outcomes), "storage": storage,
               "normalization": meta["normalization"], "zscore_reference_train_only": meta["zscore_reference_train_only"],
               "validation": None if validation is None else {
                   "passed": validation["passed"], "summary": validation["summary"],
                   "checks": [{k: c[k] for k in ("name", "passed", "n_checked", "n_failures")} for c in validation["checks"]]}}
    out = reports_dir() / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "final_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    from .plots import plot_bt_histograms, plot_context_availability
    plot_bt_histograms(edges, {s: np.asarray(c) for s, c in hist["counts"].items()},
                       reports_dir() / "figures" / "bt_histograms_by_split.png")
    plot_context_availability(events, reports_dir() / "figures" / "context_availability_by_event.png")
    return summary


def summary_markdown(s: dict) -> str:
    """A short human-readable report; every number is also in final_summary.json."""
    pct = lambda x: "–" if x is None else f"{100 * x:.2f}%"  # noqa: E731
    ev, st, t, v = s["events"], s["storage"], s["totals"], s["validation"]
    return "\n".join([
        f"# Final dataset summary: {s['dataset']}\n\nBuilt {s['created_utc']} by `sms summarize`; full details in "
        f"`final_summary.json`.\n\nTotals: {t['n_events']} events, {t['n_scans']} scans, {t['n_windows']} windows, "
        f"{t['n_patches']} patches.\n",
        _table(["Event", "Split", "Satellite", "Window (UTC)", "Scans", "Missing", "Segments", "Windows", "Patches",
                "I₋₁ / I₂ / both"],
               [[e["event_id"], e["split"], f"{e['satellite']} {e['sector']}",
                 f"{e['window_start_utc'][:16]} – {e['window_end_utc'][11:16]}", e["n_scans_processed"],
                 s["missing_data"][e["event_id"]]["missing_scans"], e["n_segments"], e.get("n_windows"), e.get("n_patches"),
                 f"{e.get('n_ctx_before', 0)} / {e.get('n_ctx_after', 0)} / {e.get('n_ctx_both', 0)}"] for e in ev]),
        "\n" + _table(["Split", "Events", "Windows", "Patches", "Mean BT (K)", "SD (K)", "p01–p99 (K)", "<220 K"],
                      [[k, ", ".join(c["events"]), c["n_windows"], c["n_patches"], f"{b['mean_K']:.2f}", f"{b['std_K']:.2f}",
                        f"{b['p01_K']:.1f}–{b['p99_K']:.1f}", pct(b["fraction_below_220K"])]
                       for k, c in s["splits"].items() for b in [s["bt_distribution"].get(k, {})] if b]),
        "\nWindow start scans: " + ", ".join(f"{k} {n}" for k, n in s["window_start_outcomes"].items()) + ".",
        f"\nStorage: segments {st['processed_segments_bytes'] / 1e9:.2f} GB ({st['bytes_per_processed_frame'] / 1e6:.2f} MB "
        f"per frame), index {st['dataset_index_bytes'] / 1e6:.2f} MB; materialised patches would need "
        f"{st['materialised_patches_uncompressed_bytes'] / 1e9:.2f} GB.",
        f"\nNormalisation: x = (T − {s['normalization']['offset_K']} K) / {s['normalization']['scale_K']} K; training-only "
        f"z-score reference {s['zscore_reference_train_only']['offset_K']:.4f} K ± {s['zscore_reference_train_only']['scale_K']:.4f} K.",
        "\nValidation: " + ("not run." if v is None else f"**{'PASS' if v['passed'] else 'FAIL'}**, "
                            f"{sum(c['passed'] for c in v['checks'])}/{len(v['checks'])} checks passed."), ""])
