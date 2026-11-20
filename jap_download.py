import os, bz2, shutil
import numpy as np
import s3fs
import matplotlib.pyplot as plt
from satpy import Scene

fs = s3fs.S3FileSystem(anon=True)
folder = "noaa-himawari9/AHI-L1b-Japan/2023/10/02/0000/"
os.makedirs("data", exist_ok=True)

# 1) download only Band 13 files, then decompress
dat_files = []
for f in sorted(fs.glob(folder + "*_B13_*")):
    bz_path = os.path.join("data", os.path.basename(f))
    dat_path = bz_path[:-4]                      # strip ".bz2"
    if not os.path.exists(dat_path):
        fs.get(f, bz_path)
        with bz2.open(bz_path, "rb") as fin, open(dat_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    dat_files.append(dat_path)
print(len(dat_files), "files ready")
assert dat_files, "No Band 13 files found. Check the folder path."   # NEW

# 2) read each one as brightness temperature (Kelvin)
for path in dat_files:
    scn = Scene(filenames=[path], reader="ahi_hsd")
    scn.load(["B13"], calibration="brightness_temperature")
    bt = scn["B13"].values
    t = scn["B13"].attrs["time_parameters"]["observation_start_time"]
    print(os.path.basename(path)[-24:], bt.shape,
          "min %.1f K, max %.1f K" % (np.nanmin(bt), np.nanmax(bt)), t)

valid = ~np.isnan(bt)
print("shape:", bt.shape, "valid fraction: %.2f" % valid.mean())

def span(v):
    i = np.where(v)[0]
    return (int(i.min()), int(i.max())) if len(i) else None

print("Upper box rows:", span(valid[:, 1300]), "cols:", span(valid[150, :]))
print("Lower box rows:", span(valid[:, 200]),  "cols:", span(valid[1050, :]))

M = 4   # trim a few pixels at the ragged edges
BOXES = {
    "A_upper": (slice(98 + M, 599),   slice(391 + M, 1447 - M)),
    "B_lower": (slice(599 + M, 1123 - M), slice(157 + M, 1208 - M)),
}
for name, (rs, cs) in BOXES.items():
    crop = bt[rs, cs]
    print(name, crop.shape,
          "NaN fraction: %.4f" % np.isnan(crop).mean(),
          "min %.1f K" % np.nanmin(crop))

# 3) quick look at the last one
plt.imshow(bt, cmap="gray_r")
plt.colorbar(label="K")
plt.show()
