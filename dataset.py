"""Phase 4: build the interpolation dataset index (windows, context frames, patches, splits, normalisation)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import __version__
from .acquire import read_csv, read_event_windows, write_csv
from .config import (REPO_ROOT, SPLITS, frames_per_gap, from_portable, load_events, load_splits, processed_dir,
                     reports_dir, split_of)
from .preprocess import j2000_to_datetime, open_segment, read_frames, read_segment, read_segment_index, valid_mask

REJECTION_REASONS = ("segment_end", "missing_minute_scan", "interval_out_of_tolerance", "endpoint_gap",
                     "hidden_time_deviation")
HIST_EDGES_K = np.arange(150.0, 345.0 + 0.25, 0.5)  # brightness-temperature histogram stored in the metadata
WINDOW_FIELDS = [
    "window_id", "split", "event_id", "segment_id", "segment_path", "gap_min", "n_hidden", "core_first_index",
    "core_last_index", "core_indices", "n_skipped_in_core", "t0_utc", "t1_utc", "span_s", "core_times_j2000",
    "t_fractions", "max_t_deviation", "ctx_before_available", "ctx_before_index", "ctx_before_time_utc",
    "ctx_before_offset_s", "ctx_before_offset_gap_units", "ctx_before_deviation_s", "ctx_after_available",
    "ctx_after_index", "ctx_after_time_utc", "ctx_after_offset_s", "ctx_after_offset_gap_units",
    "ctx_after_deviation_s", "n_patches"]
PATCH_FIELDS = ["patch_id", "window_id", "split", "event_id", "row0", "col0", "size", "kind", "core_valid_fraction",
                "cold_fraction_endpoints", "ctx_before_valid_fraction", "ctx_after_valid_fraction"]


def dataset_dir() -> Path:
    return processed_dir() / "dataset"


# ---------------------------------------------------------------- windows from real scan times
def select_core(t: np.ndarray, i: int, n: int, cadence_s: float, tol_s: float) -> list[int] | None:
    """Scans nearest t[i] + k * cadence (k = 0..n), each within tol_s of its minute mark; None if one is missing.

    With 1-minute data this picks consecutive scans; with 30-second imaging, every other scan.
    """
    idx = [i]
    for k in range(1, n + 1):
        target = t[i] + k * cadence_s
        j = int(np.searchsorted(t, target))
        best = min((c for c in (j - 1, j) if 0 <= c < len(t)), key=lambda c: abs(t[c] - target))
        if abs(t[best] - target) > tol_s or best <= idx[-1]:
            return None
        idx.append(best)
    return idx


def find_context(t: np.ndarray, target: float, tol_s: float, t0: float, span: float, allowed: np.ndarray) -> dict | None:
    """The real scan nearest a nominal context time (one gap before I0 / after I1), or None. Never synthesised."""
    if allowed.size == 0:
        return None
    j = int(allowed[np.argmin(np.abs(t[allowed] - target))])
    if abs(t[j] - target) > tol_s:
        return None
    return {"index": j, "time": float(t[j]), "offset_s": float(t[j] - t0), "offset_gap": float((t[j] - t0) / span),
            "deviation_s": float(t[j] - target)}


def evaluate_start(t: np.ndarray, i: int, gap_min: int, cadence_s: float, win: dict) -> tuple[dict | None, str]:
    """Window whose first core frame is scan i, or None with the first rule it fails.

    Normalised time of core frame k: t_k = (tau_k - tau_0) / (tau_n - tau_0), from real scan mid-times.
    """
    n, gap_s = frames_per_gap(gap_min, cadence_s), gap_min * 60.0
    if t[-1] - t[i] < gap_s - win["endpoint_gap_tolerance_s"]:
        return None, "segment_end"
    idx = select_core(t, i, n, cadence_s, win["cadence_tolerance_s"])
    if idx is None:
        return None, "missing_minute_scan"
    core = t[idx]
    if np.any(np.abs(np.diff(core) - cadence_s) > win["cadence_tolerance_s"]):
        return None, "interval_out_of_tolerance"
    span = core[-1] - core[0]
    if abs(span - gap_s) > win["endpoint_gap_tolerance_s"]:
        return None, "endpoint_gap"
    frac = (core - core[0]) / span
    dev = float(np.max(np.abs(frac - np.arange(n + 1) / n)))
    if dev > win["hidden_time_tolerance"]:
        return None, "hidden_time_deviation"
    before, after = win["context_offset_gaps"]
    return {"gap_min": gap_min, "core": [int(j) for j in idx], "times": core, "t": frac, "max_t_deviation": dev,
            "ctx_before": find_context(t, core[0] + before * gap_s, win["context_tolerance_s"], core[0], span,
                                       np.arange(0, idx[0])),
            "ctx_after": find_context(t, core[0] + after * gap_s, win["context_tolerance_s"], core[0], span,
                                      np.arange(idx[-1] + 1, len(t)))}, "accepted"


def build_windows(times: np.ndarray, cadence_s: float, win: dict) -> tuple[list[dict], dict[str, int]]:
    """Accepted windows, one every stride minutes, and the outcome count of every start scan."""
    t = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(t) <= 0):
        raise ValueError("segment times must be strictly increasing")
    windows, outcomes = [], {"accepted": 0, **dict.fromkeys(REJECTION_REASONS, 0)}
    for gap in win["gaps_min"]:
        last = -np.inf
        for i in range(len(t)):
            w, outcome = evaluate_start(t, i, gap, cadence_s, win)
            outcomes[outcome] += 1
            if w is not None and w["times"][0] - last >= win["stride_min"] * 60.0 - win["cadence_tolerance_s"]:
                windows.append(w)
                last = w["times"][0]
    return windows, outcomes


# ---------------------------------------------------------------- patches
def select_patches(core_valid: np.ndarray, bt0: np.ndarray, bt1: np.ndarray, window_id: str, pc: dict) -> list[dict]:
    """Crop positions for one window: a cold-centred crop plus seeded random crops, fully valid in all core frames.

    Chosen from the endpoint frames only (the model inputs), never from the hidden targets. Patches are
    indices, not copies: every frame of a segment shares one grid, so (row0, col0) is aligned in all frames.
    """
    H, W = core_valid.shape
    size = pc["size_px"]
    sat = np.zeros((H + 1, W + 1), dtype=np.int64)  # summed-area table: valid count of any square in O(1)
    sat[1:, 1:] = np.cumsum(np.cumsum(core_valid, axis=0), axis=1)
    with np.errstate(invalid="ignore"):
        cold = (bt0 < pc["cold_threshold_K"]) | (bt1 < pc["cold_threshold_K"])
    patches, seen = [], set()

    def try_add(r: int, c: int, kind: str) -> None:
        if (r, c) in seen:
            return
        seen.add((r, c))
        frac = float(sat[r + size, c + size] - sat[r, c + size] - sat[r + size, c] + sat[r, c]) / size**2
        if frac + 1e-12 >= pc["min_core_valid_fraction"]:
            patches.append({"row0": r, "col0": c, "size": size, "kind": kind, "core_valid_fraction": round(frac, 6),
                            "cold_fraction_endpoints": round(float(cold[r:r + size, c:c + size].mean()), 6)})

    if cold.any():
        rows, cols = np.nonzero(cold)
        r, c = int(round(rows.mean() - size / 2)), int(round(cols.mean() - size / 2))
        try_add(min(max(r, 0), H - size), min(max(c, 0), W - size), "cold_centred")
    seed = int.from_bytes(hashlib.sha256(f"{pc['seed']}:{window_id}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)  # depends only on (seed, window), not on processing order
    for _ in range(pc["max_random_tries"]):
        if len(patches) >= pc["per_window"]:
            break
        try_add(int(rng.integers(0, H - size + 1)), int(rng.integers(0, W - size + 1)), "random")
    return patches


# ---------------------------------------------------------------- normalisation
def normalize(bt_K: np.ndarray, offset_K: float, scale_K: float) -> np.ndarray:
    """x = (T - offset) / scale; exactly invertible, never clipped (overshooting tops can be colder than 180 K)."""
    bt = np.asarray(bt_K)
    f = bt.dtype.type if bt.dtype.kind == "f" else np.float64
    return (bt - f(offset_K)) / f(scale_K)


def denormalize(x: np.ndarray, offset_K: float, scale_K: float) -> np.ndarray:
    x = np.asarray(x)
    f = x.dtype.type if x.dtype.kind == "f" else np.float64
    return x * f(scale_K) + f(offset_K)


def new_moments() -> dict:
    return {"n": 0, "mean": 0.0, "m2": 0.0, "min": math.inf, "max": -math.inf, "below": 0, "above": 0}


def update_moments(m: dict, values: np.ndarray, lo: float = -math.inf, hi: float = math.inf) -> None:
    """Streaming mean/variance over hundreds of millions of pixels, one frame at a time (Chan et al., 1983).

    Merges the frame's mean and sum of squared deviations into the running totals, which keeps full
    precision; accumulating sum(x) and sum(x^2) in float32 does not (see PDF 04 for the measured failure).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return
    nb, mb = v.size, float(v.mean())
    m2b, delta, tot = float(((v - mb) ** 2).sum()), mb - m["mean"], m["n"] + v.size
    m["mean"] += delta * nb / tot
    m["m2"] += m2b + delta * delta * m["n"] * nb / tot
    m["n"] = tot
    m["min"], m["max"] = min(m["min"], float(v.min())), max(m["max"], float(v.max()))
    m["below"] += int(np.count_nonzero(v < lo))
    m["above"] += int(np.count_nonzero(v > hi))


def std_of(m: dict) -> float:
    return math.sqrt(m["m2"] / m["n"]) if m["n"] else float("nan")


# ---------------------------------------------------------------- leakage checks
def temporal_separation_issues(owner: dict[str, str], intervals: dict, min_hours: float) -> list[str]:
    """Events in different splits must be >= min_hours apart (one weather system stays on one side)."""
    issues, ev = [], sorted(intervals)
    for i, a in enumerate(ev):
        for b in ev[i + 1:]:
            if owner.get(a) and owner.get(b) and owner[a] != owner[b]:
                (a0, a1), (b0, b1) = intervals[a], intervals[b]
                hours = max((b0 - a1).total_seconds(), (a0 - b1).total_seconds()) / 3600.0
                if hours < min_hours:
                    issues.append(f"{a} ({owner[a]}) and {b} ({owner[b]}) are only {hours:.1f} h apart")
    return issues


def shared_source_issues(items_by_split: dict[str, set]) -> list[str]:
    """No source file (or file content) may be used by two splits."""
    issues, owner = [], {}
    for split, items in items_by_split.items():
        for k in items:
            if k in owner and owner[k] != split:
                issues.append(f"{k} is used by both {owner[k]} and {split}")
            owner.setdefault(k, split)
    return issues


# ---------------------------------------------------------------- building the index (sms build-dataset)
def _iso(seconds: float) -> str:
    return j2000_to_datetime(seconds).isoformat(timespec="milliseconds")


def _window_row(w: dict, window_id: str, split: str, seg: dict, n_patches: int) -> dict:
    row = {"window_id": window_id, "split": split, "event_id": seg["event_id"], "segment_id": seg["segment_id"],
           "segment_path": seg["path"], "gap_min": w["gap_min"], "n_hidden": len(w["core"]) - 2,
           "core_first_index": w["core"][0], "core_last_index": w["core"][-1],
           "core_indices": ";".join(map(str, w["core"])),
           "n_skipped_in_core": w["core"][-1] - w["core"][0] + 1 - len(w["core"]),
           "t0_utc": _iso(w["times"][0]), "t1_utc": _iso(w["times"][-1]), "span_s": f"{w['times'][-1] - w['times'][0]:.3f}",
           "core_times_j2000": ";".join(f"{t:.3f}" for t in w["times"]), "t_fractions": ";".join(f"{t:.6f}" for t in w["t"]),
           "max_t_deviation": f"{w['max_t_deviation']:.6f}", "n_patches": n_patches}
    for name in ("before", "after"):
        c = w[f"ctx_{name}"]
        row |= ({f"ctx_{name}_available": 0, f"ctx_{name}_index": -1, f"ctx_{name}_time_utc": "", f"ctx_{name}_offset_s": "",
                 f"ctx_{name}_offset_gap_units": "", f"ctx_{name}_deviation_s": ""} if c is None else
                {f"ctx_{name}_available": 1, f"ctx_{name}_index": c["index"], f"ctx_{name}_time_utc": _iso(c["time"]),
                 f"ctx_{name}_offset_s": f"{c['offset_s']:.3f}", f"ctx_{name}_offset_gap_units": f"{c['offset_gap']:.6f}",
                 f"ctx_{name}_deviation_s": f"{c['deviation_s']:.3f}"})
    return row


def build_dataset(cfg: dict) -> dict:
    """Windows, patches, split checks and pixel statistics for every processed segment of every assigned event."""
    owner = split_of(load_splits(REPO_ROOT / cfg["splits"]["file"], load_events()["events"]))
    segments = read_segment_index()
    if not segments:
        raise RuntimeError("no processed segments; run `sms process` first")
    cadence, usable, pc = cfg["instrument"]["nominal_cadence_s"], cfg["quality"]["usable_dqf"], cfg["patches"]
    lo = cfg["normalization"]["offset_K"]
    hi = lo + cfg["normalization"]["scale_K"]
    window_rows, patch_rows, outcomes, skipped = [], [], {}, set()
    sources = {s: set() for s in SPLITS}
    moments = {s: new_moments() for s in SPLITS}
    hist = {s: np.zeros(HIST_EDGES_K.size - 1, dtype=np.int64) for s in SPLITS}
    hist_outside = {s: [0, 0] for s in SPLITS}
    counted = set()  # (segment, frame): each frame enters the pixel statistics once

    for seg in segments:
        event, path = seg["event_id"], from_portable(seg["path"])
        if event not in owner:
            skipped.add(event)
            continue
        split, meta = owner[event], read_segment(path)
        windows, counts = build_windows(meta["time_mid"], cadence, cfg["windows"])
        ds = open_segment(path)
        for k, v in counts.items():
            outcomes.setdefault(event, {})
            outcomes[event][k] = outcomes[event].get(k, 0) + v
        for w in windows:
            window_id = f"{seg['segment_id']}_g{w['gap_min']:02d}_i{w['core'][0]:04d}"
            bt, dqf = read_frames(ds, w["core"])
            valid = valid_mask(bt, dqf, usable)
            patches = select_patches(valid.all(axis=0), bt[0], bt[-1], window_id, pc)
            ctx_valid = {}
            for name in ("before", "after"):
                c = w[f"ctx_{name}"]
                ctx_valid[name] = None if c is None else valid_mask(*read_frames(ds, c["index"]), usable)
            for p_i, p in enumerate(patches):
                rs, cs = slice(p["row0"], p["row0"] + p["size"]), slice(p["col0"], p["col0"] + p["size"])
                patch_rows.append({"patch_id": f"{window_id}_p{p_i}", "window_id": window_id, "split": split,
                                   "event_id": event, **p,
                                   **{f"ctx_{n}_valid_fraction": "" if m is None else round(float(m[rs, cs].mean()), 6)
                                      for n, m in ctx_valid.items()}})
            window_rows.append(_window_row(w, window_id, split, seg, len(patches)))
            used = w["core"] + [w[f"ctx_{n}"]["index"] for n in ("before", "after") if w[f"ctx_{n}"]]
            sources[split].update(meta["source_keys"][k] for k in used)
            for j, k in enumerate(w["core"]):
                if (seg["segment_id"], k) not in counted:
                    counted.add((seg["segment_id"], k))
                    vals = bt[j][valid[j]]
                    update_moments(moments[split], vals, lo, hi)
                    hist[split] += np.histogram(vals, bins=HIST_EDGES_K)[0]
                    hist_outside[split][0] += int(np.count_nonzero(vals < HIST_EDGES_K[0]))
                    hist_outside[split][1] += int(np.count_nonzero(vals > HIST_EDGES_K[-1]))
        ds.close()

    intervals = {e: (w["start"], w["end"]) for e, w in read_event_windows().items() if e in owner}
    issues = temporal_separation_issues(owner, intervals, cfg["splits"]["min_event_separation_hours"])
    issues += shared_source_issues(sources)
    if issues:
        raise RuntimeError("data leakage detected:\n  " + "\n  ".join(issues))

    write_csv(dataset_dir() / "windows.csv", WINDOW_FIELDS, window_rows)
    write_csv(dataset_dir() / "patches.csv", PATCH_FIELDS, patch_rows)
    train = moments["train"]
    train_events = sorted({r["event_id"] for r in window_rows if r["split"] == "train"})
    meta = {
        "dataset": cfg["dataset"]["name"], "processing_version": cfg["dataset"]["processing_version"],
        "software_version": __version__, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "band": cfg["instrument"]["bands"][0], "units": "K",
        "storage": "brightness temperature in per-segment NetCDF-4 files; this index references them",
        "spec": {"gaps_min": cfg["windows"]["gaps_min"], "patch_size_px": pc["size_px"],
                 "context_offset_gaps": cfg["windows"]["context_offset_gaps"], "stride_min": cfg["windows"]["stride_min"],
                 "usable_dqf": usable},
        "normalization": {"method": "fixed_affine", "offset_K": float(lo), "scale_K": float(cfg["normalization"]["scale_K"]),
                          "fitted_on": [], "formula": "x = (T - offset_K) / scale_K"},
        "zscore_reference_train_only": None if not train["n"] else {
            "method": "zscore", "offset_K": train["mean"], "scale_K": std_of(train), "fitted_on": train_events,
            "formula": "x = (T - offset_K) / scale_K"},
        "splits": {s: split_counts(window_rows, s) for s in SPLITS},
        "split_assignment": {s: [e for e, o in owner.items() if o == s] for s in SPLITS},
        "pixel_statistics_core_frames": {s: moment_summary(m, lo, hi) for s, m in moments.items()},
        "bt_histogram_core_frames": {"bin_edges_K": [float(e) for e in HIST_EDGES_K],
                                     "counts": {s: h.tolist() for s, h in hist.items()},
                                     "outside_range_below_above": hist_outside},
        "window_start_outcomes": outcomes, "per_event": per_event_counts(window_rows),
        "leakage_checks": "passed", "events_processed_but_unassigned": sorted(skipped),
        "n_windows": len(window_rows), "n_patches": len(patch_rows),
    }
    for path in (dataset_dir() / "dataset_meta.json", reports_dir() / "dataset" / "dataset_summary.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def split_counts(rows: list[dict], split: str) -> dict:
    ws = [r for r in rows if r["split"] == split]
    b = [int(r["ctx_before_available"]) for r in ws]
    a = [int(r["ctx_after_available"]) for r in ws]
    return {"events": sorted({r["event_id"] for r in ws}), "n_windows": len(ws),
            "n_patches": sum(int(r["n_patches"]) for r in ws),
            "n_windows_without_patch": sum(int(r["n_patches"]) == 0 for r in ws),
            "n_ctx_before": sum(b), "n_ctx_after": sum(a), "n_ctx_both": sum(x and y for x, y in zip(b, a))}


def per_event_counts(rows: list[dict]) -> dict:
    out = {}
    for e in dict.fromkeys(r["event_id"] for r in rows):
        ws = [r for r in rows if r["event_id"] == e]
        c = split_counts(ws, ws[0]["split"])
        out[e] = {"split": ws[0]["split"], "n_windows": c["n_windows"], "n_patches": c["n_patches"],
                  "n_windows_without_patch": c["n_windows_without_patch"], "n_ctx_before": c["n_ctx_before"],
                  "n_ctx_after": c["n_ctx_after"], "n_ctx_both": c["n_ctx_both"],
                  "n_windows_with_skipped_scans": sum(int(r["n_skipped_in_core"]) > 0 for r in ws)}
    return out


def moment_summary(m: dict, lo: float, hi: float) -> dict:
    if not m["n"]:
        return {"n_pixels": 0}
    return {"n_pixels": m["n"], "mean_K": m["mean"], "std_K": std_of(m), "min_K": m["min"], "max_K": m["max"],
            f"fraction_below_{lo:g}K": m["below"] / m["n"], f"fraction_above_{hi:g}K": m["above"] / m["n"]}


# ---------------------------------------------------------------- reading the dataset
def load_index(split: str | None = None, allow_test: bool = False) -> tuple[list[dict], list[dict]]:
    """Window and patch rows. Test rows are released only with allow_test=True (final evaluation)."""
    if split not in (None, *SPLITS):
        raise ValueError(f"unknown split {split!r}")
    if split == "test" and not allow_test:
        raise PermissionError("test windows are reserved for the final evaluation; pass allow_test=True")
    windows, patches = read_csv(dataset_dir() / "windows.csv"), read_csv(dataset_dir() / "patches.csv")

    def keep(r):
        return r["split"] == split if split else (allow_test or r["split"] != "test")
    return [w for w in windows if keep(w)], [p for p in patches if keep(p)]


def core_indices(window: dict) -> list[int]:
    return [int(i) for i in window["core_indices"].split(";")]


def read_sample(window: dict, patch: dict, usable_dqf=(0,)) -> dict:
    """One window cropped to one patch, in kelvin: core frames (I0, hidden..., I1), validity, t, context frames."""
    r0, c0, size = int(patch["row0"]), int(patch["col0"]), int(patch["size"])
    rs, cs = slice(r0, r0 + size), slice(c0, c0 + size)
    path = from_portable(window["segment_path"])
    bt, dqf = read_frames(path, core_indices(window), rs, cs)
    sample = {"window_id": window["window_id"], "patch_id": patch["patch_id"], "core_bt_K": bt,
              "core_valid": valid_mask(bt, dqf, usable_dqf),
              "t_fractions": np.array([float(t) for t in window["t_fractions"].split(";")])}
    for name in ("before", "after"):
        have = window[f"ctx_{name}_available"] == "1"
        sample[f"ctx_{name}_bt_K"] = read_frames(path, int(window[f"ctx_{name}_index"]), rs, cs)[0] if have else None
        sample[f"ctx_{name}_t"] = float(window[f"ctx_{name}_offset_gap_units"]) if have else None
    return sample
