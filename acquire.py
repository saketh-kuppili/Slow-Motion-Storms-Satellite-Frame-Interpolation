"""Phase 2: find GOES ABI files and download them with checksums (event selection: scripts/select_events.py)."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import re
from pathlib import Path

import numpy as np
import s3fs

from .config import UTC, from_portable, manifest_dir, raw_dir, to_portable

MANIFEST_FIELDS = ["purpose", "event_id", "satellite", "bucket", "key", "product", "scene", "channel",
                   "scan_start_utc", "scan_end_utc", "size_bytes", "sha256", "local_path"]
_ABI_NAME = re.compile(r"OR_ABI-(?P<level>L1b|L2)-(?P<product>Rad|CMIP)(?P<scene>M1|M2|F|C)-M(?P<mode>\d)"
                       r"C(?P<channel>\d{2})_G(?P<platform>\d{2})_s(?P<start>\d{14})_e(?P<end>\d{14})_c\d{14}\.nc$")
_fs = None


def s3() -> s3fs.S3FileSystem:
    """Anonymous access to NOAA's public buckets (no credentials needed)."""
    global _fs
    if _fs is None:
        _fs = s3fs.S3FileSystem(anon=True)
    return _fs


def read_csv(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- ABI file names and the S3 archive
def abi_time(stamp: str) -> dt.datetime:
    """``YYYYJJJHHMMSSs`` (year, day of year, time, tenths of a second) as UTC."""
    t = dt.datetime.strptime(stamp[:13], "%Y%j%H%M%S").replace(tzinfo=UTC)
    return t + dt.timedelta(milliseconds=100 * int(stamp[13]))


def parse_abi_name(name: str) -> dict | None:
    """Fields of an ABI file name, e.g. ``OR_ABI-L1b-RadM1-M6C13_G19_s2025136...nc``; None if not ABI."""
    m = _ABI_NAME.search(name.rsplit("/", 1)[-1])
    if m is None:
        return None
    g = m.groupdict()
    return {"product": g["product"], "scene": g["scene"], "channel": int(g["channel"]),
            "satellite": f"goes{g['platform']}", "start_stamp": g["start"],
            "start": abi_time(g["start"]), "end": abi_time(g["end"])}


def list_abi_files(satellite: str, product: str, start: dt.datetime, end: dt.datetime,
                   scene: str | None = None, channel: int | None = None) -> list[dict]:
    """Files whose scan starts in [start, end), from ``noaa-<satellite>/<product>/<year>/<doy>/<hour>/``."""
    bucket, files = f"noaa-{satellite}", []
    hour = start.replace(minute=0, second=0, microsecond=0)
    while hour < end:
        prefix = f"{bucket}/{product}/{hour:%Y}/{hour.timetuple().tm_yday:03d}/{hour:%H}/"
        try:
            listing = s3().ls(prefix, detail=True, refresh=True)
        except FileNotFoundError:
            listing = []
        for obj in listing:
            meta = parse_abi_name(obj["name"])
            if (meta and (scene is None or meta["scene"] == scene) and (channel is None or meta["channel"] == channel)
                    and start <= meta["start"] < end):
                files.append({**meta, "bucket": bucket, "key": obj["name"].split("/", 1)[1], "size": obj["size"]})
        hour += dt.timedelta(hours=1)
    return sorted(files, key=lambda f: (f["start"], f["key"]))


# ---------------------------------------------------------------- downloads and the manifest
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(f: dict, purpose: str, event_id: str) -> dict:
    """Download one archive file to ``data/raw/<bucket>/<key>`` (reused if the size matches); return its manifest row."""
    dest = raw_dir() / f["bucket"] / f["key"]
    if not (dest.exists() and dest.stat().st_size == f["size"]):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")  # an interrupted download never looks complete
        s3().get(f"{f['bucket']}/{f['key']}", str(tmp))
        if tmp.stat().st_size != f["size"]:
            tmp.unlink()
            raise IOError(f"size mismatch for {f['key']}")
        tmp.replace(dest)
    return {"purpose": purpose, "event_id": event_id, "satellite": f["satellite"], "bucket": f["bucket"],
            "key": f["key"], "product": f["product"], "scene": f["scene"], "channel": f["channel"],
            "scan_start_utc": f["start"].isoformat(timespec="milliseconds"),
            "scan_end_utc": f["end"].isoformat(timespec="milliseconds"),
            "size_bytes": f["size"], "sha256": sha256_file(dest), "local_path": to_portable(dest)}


def read_manifest() -> list[dict]:
    rows = read_csv(manifest_dir() / "manifest.csv")
    for r in rows:
        r["channel"], r["size_bytes"] = int(r["channel"]), int(r["size_bytes"])
    return rows


def merge_into_manifest(new_rows: list[dict]) -> None:
    """One row per (purpose, event, key), sorted; new rows replace old ones."""
    rows = {(r["purpose"], r["event_id"], r["key"]): r for r in read_manifest() + list(new_rows)}
    write_csv(manifest_dir() / "manifest.csv", MANIFEST_FIELDS, [rows[k] for k in sorted(rows)])


def verify_manifest(rows: list[dict]) -> list[str]:
    problems = []
    for r in rows:
        local = from_portable(r["local_path"])
        if not local.exists():
            problems.append(f"missing: {r['local_path']}")
        elif sha256_file(local) != r["sha256"]:
            problems.append(f"checksum mismatch: {r['local_path']}")
    return problems


# ---------------------------------------------------------------- derived event windows and tracks (read back)
def interpolate_track(track: list[dict], when: dt.datetime) -> tuple[float, float]:
    """Linear interpolation of latitude/longitude in time (clamped at the ends)."""
    pts = sorted(track, key=lambda p: p["time"])
    ts = [p["time"].timestamp() for p in pts]
    return (float(np.interp(when.timestamp(), ts, [p["lat"] for p in pts])),
            float(np.interp(when.timestamp(), ts, [p["lon"] for p in pts])))


def read_event_windows() -> dict[str, dict]:
    out = {}
    for r in read_csv(manifest_dir() / "event_windows.csv"):
        tags = dict(item.split("=", 1) for item in r["tags"].split(";") if item)
        out[r["event_id"]] = {"event_id": r["event_id"], "name": r["name"], "type": r["type"],
                              "satellite": r["satellite"], "start": dt.datetime.fromisoformat(r["window_start_utc"]),
                              "end": dt.datetime.fromisoformat(r["window_end_utc"]), "rule": r["rule"], "tags": tags}
    return out


def read_event_tracks() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in read_csv(manifest_dir() / "event_tracks.csv"):
        out.setdefault(r["event_id"], []).append({
            "time": dt.datetime.fromisoformat(r["time_utc"]), "lat": float(r["lat"]), "lon": float(r["lon"]),
            "wind_kt": float(r["wind_kt"]) if r["wind_kt"] else None,
            "n_reports": int(r["n_reports"]) if r["n_reports"] else None})
    return out


# ---------------------------------------------------------------- downloading verified events
def download_event(event_id: str, cfg: dict, offset_min: float | None = None, duration_min: float | None = None,
                   with_references: bool = True) -> list[dict]:
    """All Band 13 scans of a verified event (optionally a sub-interval) plus every n-th NOAA CMIP reference."""
    sel = {r["event_id"]: r for r in read_csv(manifest_dir() / "event_selection.csv")}.get(event_id)
    if sel is None or sel["status"] != "pass":
        raise ValueError(f"{event_id} has not passed coverage verification")
    start, end = dt.datetime.fromisoformat(sel["window_start_utc"]), dt.datetime.fromisoformat(sel["window_end_utc"])
    if offset_min is not None:
        start += dt.timedelta(minutes=offset_min)
    if duration_min is not None:
        end = min(end, start + dt.timedelta(minutes=duration_min))
    inst, rows = cfg["instrument"], []
    for band in inst["bands"]:
        files = list_abi_files(sel["satellite"], inst["product"], start, end, scene=sel["sector"], channel=band)
        rows += [fetch(f, "dataset", event_id) for f in files]
        if with_references:
            wanted = {f["start_stamp"] for f in files[:: cfg["calibration"]["check_every_nth_scan"]]}
            refs = list_abi_files(sel["satellite"], inst["reference_product"], start, end, scene=sel["sector"], channel=band)
            rows += [fetch(r, "calibration_reference", event_id) for r in refs if r["start_stamp"] in wanted]
    merge_into_manifest(rows)
    return rows
