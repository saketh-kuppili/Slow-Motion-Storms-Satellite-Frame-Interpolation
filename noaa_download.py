import os, datetime as dt
import numpy as np
import s3fs
import matplotlib.pyplot as plt
from satpy import Scene

# ---------- settings ----------
REGION  = "east"               # "east", "west", or "both"
DATE    = dt.date(2023, 6, 2)
HOUR    = 21                   # UTC
SECTORS = ["M1"]               # ["M1"], ["M2"], or ["M1", "M2"]
N_FILES = 4                    # consecutive 1-minute files to test
# ------------------------------
assert REGION in ("east", "west", "both"), "REGION must be east, west, or both"

def sats_for(day):
    east = "goes16" if day < dt.date(2025, 4, 7) else "goes19"
    if REGION == "east":
        return [east]
    if REGION == "west":
        return ["goes18"]
    return [east, "goes18"]

fs = s3fs.S3FileSystem(anon=True)
doy = DATE.timetuple().tm_yday
os.makedirs("data_goes", exist_ok=True)

last = {}   # (satellite, sector) -> (last frame, its time), for plotting

for sat in sats_for(DATE):
    folder = f"noaa-{sat}/ABI-L1b-RadM/{DATE.year}/{doy:03d}/{HOUR:02d}/"
    print("\nfolder:", folder)
    try:
        names = fs.ls(folder)
    except FileNotFoundError:
        names = []
    if not names:
        print(sat, "has no data for this date/hour")
        continue

    for sec in SECTORS:
        # 1) keep only Band 13 of this sector, download the first few files
        b13 = sorted(n for n in names
                     if f"RadM{sec[1]}-" in n and "C13_G" in n)
        print(sat, sec, "B13 files:", len(b13))
        if not b13:
            continue

        # 2) read each one as brightness temperature (Kelvin)
        times = []
        for f in b13[:N_FILES]:
            local = os.path.join("data_goes", os.path.basename(f))
            if not os.path.exists(local):
                fs.get(f, local)
            scn = Scene(filenames=[local], reader="abi_l1b")
            scn.load(["C13"], calibration="brightness_temperature")
            da = scn["C13"]
            bt = da.values
            t = da.attrs["start_time"]
            times.append(t)
            lon, lat = da.attrs["area"].get_lonlat(bt.shape[0] // 2,
                                                   bt.shape[1] // 2)
            print(" ", sat, sec, bt.shape,
                  "min %.1f K, max %.1f K" % (np.nanmin(bt), np.nanmax(bt)),
                  t, "center (%.2f, %.2f)" % (lat, lon))

        for a, b in zip(times[:-1], times[1:]):
            print("  gap between files: %.0f s" % (b - a).total_seconds())
        print("  valid fraction: %.3f" % (~np.isnan(bt)).mean())
        last[(sat, sec)] = (bt, times[-1])

assert last, "Nothing was read. Check REGION, DATE, HOUR and SECTORS."

# 3) quick look at the last frame of each satellite/sector
fig, axes = plt.subplots(1, len(last), figsize=(6 * len(last), 5),
                         squeeze=False)
for ax, ((sat, sec), (bt, t)) in zip(axes[0], last.items()):
    im = ax.imshow(bt, cmap="gray_r")
    ax.set_title(f"{sat} {sec}\n{t}")
    plt.colorbar(im, ax=ax, label="K")
plt.tight_layout()
plt.show()