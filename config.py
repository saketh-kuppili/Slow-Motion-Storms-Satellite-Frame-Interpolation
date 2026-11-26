"""Phase 1: paths, configuration and the experimental-specification check."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import yaml

UTC = dt.timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
# Last day with GOES-16 mesoscale files in noaa-goes16 (bucket listing); GOES-19 afterwards.
GOES16_LAST_EAST_DAY = dt.date(2025, 4, 7)


# ---------------------------------------------------------------- paths
def _env_dir(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def data_dir() -> Path:  # downloads and processed data (git-ignored); relocatable for slow drives
    return _env_dir("SMS_DATA_DIR", REPO_ROOT / "data")


def raw_dir() -> Path:
    return data_dir() / "raw"


def processed_dir() -> Path:
    return data_dir() / "processed"


def manifest_dir() -> Path:  # committed manifests; overridable so tests can run in a temp dir
    return _env_dir("SMS_MANIFEST_DIR", REPO_ROOT / "data" / "manifest")


def reports_dir() -> Path:
    return _env_dir("SMS_REPORTS_DIR", REPO_ROOT / "reports")


def configs_dir() -> Path:
    return REPO_ROOT / "configs"


def sample_dir() -> Path:
    return REPO_ROOT / "data" / "sample"


def to_portable(path: Path, base: Path | None = None) -> str:
    """Path relative to the data directory, POSIX style, as stored in manifests."""
    base = data_dir() if base is None else base
    return Path(os.path.relpath(Path(path).resolve(), base.resolve())).as_posix()


def from_portable(rel: str, base: Path | None = None) -> Path:
    base = data_dir() if base is None else base
    return (base / rel).resolve()


# ---------------------------------------------------------------- configuration
def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: Path | None = None) -> dict:
    """The dataset specification ``configs/data/base.yaml`` as nested dictionaries."""
    return load_yaml(path or configs_dir() / "data" / "base.yaml")


def parse_utc(value) -> dt.datetime:
    """YAML date, datetime or ISO string (e.g. ``2021-08-29T00:00Z``) as an aware UTC datetime."""
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    elif not isinstance(value, dt.datetime):
        value = dt.datetime(value.year, value.month, value.day)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def load_events(path: Path | None = None) -> dict:
    """``configs/data/events.yaml``: ``{"sources": {...}, "events": {id: event}}`` (candidates first, then backups)."""
    raw = load_yaml(path or configs_dir() / "data" / "events.yaml")
    events = {}
    for group in ("events", "backups"):
        for ev in raw.get(group, []):
            track = dict(ev["track"])
            for key in ("search_start", "search_end"):
                if key in track:
                    track[key] = parse_utc(track[key])
            if "convective_day" in track:
                track["convective_day"] = parse_utc(track["convective_day"]).date()
            if ev["id"] in events:
                raise ValueError(f"duplicate event id {ev['id']}")
            events[ev["id"]] = {**ev, "track": track, "is_backup": group == "backups"}
    return {"sources": raw["sources"], "events": events}


def event_day(ev: dict) -> dt.date:
    t = ev["track"]
    return t["convective_day"] if t["source"] == "spc" else t["search_start"].date()


def load_splits(path: Path | None = None, known_events=None) -> dict[str, list[str]]:
    """Event-level split file; every event in at most one split and known to the catalogue."""
    raw = load_yaml(path or configs_dir() / "data" / "splits.yaml")
    if set(raw) - set(SPLITS):
        raise ValueError(f"unknown split names {sorted(set(raw) - set(SPLITS))}")
    splits = {s: list(raw.get(s) or []) for s in SPLITS}
    owner = {}
    for s, events in splits.items():
        for e in events:
            if known_events is not None and e not in known_events:
                raise ValueError(f"{s}: unknown event {e}")
            if e in owner:
                raise ValueError(f"event {e} is in both {owner[e]} and {s}")
            owner[e] = s
    return splits


def split_of(splits: dict[str, list[str]]) -> dict[str, str]:
    return {e: s for s, events in splits.items() for e in events}


def goes_east_satellite(day: dt.date) -> str:
    return "goes16" if day <= GOES16_LAST_EAST_DAY else "goes19"


def frames_per_gap(gap_min: int, cadence_s: float) -> int:
    """Cadence steps n in a gap: the core has n + 1 frames, n - 1 of them hidden."""
    steps = gap_min * 60.0 / cadence_s
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"gap of {gap_min} min is not a whole number of {cadence_s} s steps")
    return int(round(steps))


# ---------------------------------------------------------------- specification check (sms spec)
def resolve_spec(cfg: dict) -> dict:
    """Derive the experimental design from the configs and check it is self-consistent."""
    problems = []
    research = load_yaml(configs_dir() / "research.yaml")
    events = load_events()["events"]
    try:
        splits = load_splits(REPO_ROOT / cfg["splits"]["file"], events)
    except ValueError as exc:
        problems.append(str(exc))
        splits = load_yaml(REPO_ROOT / cfg["splits"]["file"])
    owner = split_of(splits)
    win, cadence = cfg["windows"], cfg["instrument"]["nominal_cadence_s"]

    gaps = []
    for g in win["gaps_min"]:
        n = frames_per_gap(g, cadence)
        gaps.append({"gap_min": g, "steps": n, "core_frames": n + 1, "hidden_frames": n - 1,
                     "nominal_t": [round(k / n, 6) for k in range(n + 1)]})
    if not any(g["gap_min"] == 10 and g["hidden_frames"] == 9 for g in gaps):
        problems.append("the primary 10-minute gap with nine hidden frames is missing")
    before, after = win["context_offset_gaps"]
    if not (before < 0 and after > 1):
        problems.append(f"context offsets {before}, {after} must lie outside [0, 1]")
    if win["context_tolerance_s"] >= cadence:
        problems.append("context tolerance must be smaller than one cadence step")
    if win["cadence_tolerance_s"] >= cadence / 2:  # otherwise a neighbouring scan could replace a missing one
        problems.append("cadence tolerance must be below half a cadence step")
    H, W = cfg["quality"]["expected_shape_px"]
    if cfg["patches"]["size_px"] > min(H, W):
        problems.append("patch larger than the sector")
    if cfg["coverage"]["edge_margin_px"] * 2 >= min(H, W):
        problems.append("coverage edge margin leaves no interior")
    norm = cfg["normalization"]
    if norm["method"] not in ("fixed_affine", "zscore") or norm["report_zscore_from_split"] != "train":
        problems.append("normalisation must be fixed_affine or zscore, with statistics from the training split")

    rows = []
    for eid, ev in events.items():
        day = event_day(ev)
        if ev["satellite"] != goes_east_satellite(day):
            problems.append(f"{eid}: satellite {ev['satellite']} but GOES-East on {day} was {goes_east_satellite(day)}")
        rows.append({"id": eid, "name": ev["name"], "type": ev["type"], "satellite": ev["satellite"],
                     "track_source": ev["track"]["source"], "date": str(day),
                     "split": owner.get(eid, "backup" if ev["is_backup"] else "unassigned")})
    for s in SPLITS:
        if not splits.get(s):
            problems.append(f"split {s} is empty")
    train_types = {r["type"] for r in rows if r["split"] == "train"}
    test_types = {r["type"] for r in rows if r["split"] == "test"}
    if not test_types <= train_types:
        problems.append(f"test storm types {sorted(test_types - train_types)} never occur in training")
    if problems:
        raise ValueError("specification problems:\n  " + "\n  ".join(problems))

    inst, lo = cfg["instrument"], norm["offset_K"]
    return {
        "research": research,
        "dataset": cfg["dataset"],
        "instrument": {k: inst[k] for k in ("product", "reference_product", "bands", "sectors")}
                      | {"nominal_cadence_s": cadence, "orbital_slot": inst["orbital_slot"]},
        "windows": {"gaps": gaps, "context_offset_gaps": [before, after],
                    "context_nominal_offsets_min": {g["gap_min"]: [before * g["gap_min"], after * g["gap_min"]] for g in gaps},
                    "stride_min": win["stride_min"],
                    "tolerances": {"cadence_s": win["cadence_tolerance_s"], "endpoint_gap_s": win["endpoint_gap_tolerance_s"],
                                   "hidden_t": win["hidden_time_tolerance"], "context_s": win["context_tolerance_s"]}},
        "patches": {k: cfg["patches"][k] for k in ("size_px", "per_window", "min_core_valid_fraction")},
        "normalization": {"method": norm["method"], "formula": "x = (T - offset_K) / scale_K",
                          "inverse": "T = offset_K + scale_K * x", "offset_K": lo, "scale_K": norm["scale_K"],
                          "maps_K_to_unit_interval": [lo, lo + norm["scale_K"]], "clipping": "none"},
        "quality": {"usable_dqf": cfg["quality"]["usable_dqf"],
                    "calibration_tolerance_K": cfg["calibration"]["max_abs_diff_vs_noaa_K"]},
        "events": rows,
        "splits": {s: list(splits.get(s) or []) for s in SPLITS},
    }


def write_spec(spec: dict) -> Path:
    path = reports_dir() / "phase1" / "experiment_spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")
    return path
