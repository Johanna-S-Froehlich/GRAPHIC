"""Download WOS-46985 (Web of Science, 134 classes) for the CNN and GCN pipelines.

Pulls the WebOfScience.zip from the Mendeley dataset (CC BY 4.0) and extracts
the WOS46985 subset to `wos46985/` at the repo root:

    wos46985/X.txt    one document per line (abstracts of scientific papers)
    wos46985/Y.txt    one integer label per line, range 0..133

If you use this dataset, please cite:

    K. Kowsari, D. E. Brown, M. Heidarysafa, K. Jafari Meimandi, M. S. Gerber,
    and L. E. Barnes. "HDLTex: Hierarchical Deep Learning for Text
    Classification." ICMLA, 2017.

Note: Y.txt has 134 unique numeric labels but Classes.txt (in Data.xlsx) has
143 unique string names — 8 indices have multiple name variants. We use 134
as the canonical class count.
"""

import os
import shutil
import urllib.request
import zipfile

ZIP_URL = (
    "https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/"
    "files/c9ea673d-5542-44c0-ab7b-f1311f7d61df/file_downloaded"
)
ZIP_PATH = "WebOfScience.zip"
OUT_DIR = "wos46985"
SUBSET = "WOS46985"

# Mendeley's CDN rejects the default ``Python-urllib/<ver>`` User-Agent with
# 403 Forbidden. Sending a browser-style UA gets through.
_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) GRAPHIC-download-script"


def _download(url: str, dest: str) -> None:
    """Stream-download ``url`` to ``dest`` with a browser-style User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)


def main():
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading WebOfScience.zip from Mendeley ({ZIP_URL})")
        _download(ZIP_URL, ZIP_PATH)
        print(f"Saved to {ZIP_PATH}")
    else:
        print(f"{ZIP_PATH} already present; skipping download.")

    os.makedirs(OUT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for name in ("X.txt", "Y.txt"):
            src = f"{SUBSET}/{name}"
            dst = os.path.join(OUT_DIR, name)
            with zf.open(src) as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
            print(f"Extracted {src} -> {dst}")

    with open(os.path.join(OUT_DIR, "Y.txt")) as f:
        labels = [int(x) for x in f.read().split()]
    n_docs = len(labels)
    n_classes = len(set(labels))
    print(f"  {n_docs} documents, {n_classes} classes "
          f"(min={min(labels)}, max={max(labels)})")


if __name__ == "__main__":
    main()
