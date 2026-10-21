"""Download the FlockLab link-quality data used in exp_flocklab.py (Jacob et al., DATA'19,
doi:10.5281/zenodo.3731498, CC-BY-4.0) into ../data/flocklab/."""
import os, urllib.request

FILES = ["2019-07_preprocessed_all.zip", "2019-08_preprocessed_all.zip", "2019-09_preprocessed_all.zip"]
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "flocklab")
os.makedirs(OUT, exist_ok=True)
for f in FILES:
    path = os.path.join(OUT, f)
    if os.path.exists(path):
        print("present", f); continue
    url = f"https://zenodo.org/records/3731498/files/{f}?download=1"
    print("downloading", f)
    urllib.request.urlretrieve(url, path + ".part"); os.replace(path + ".part", path)
print("done")
