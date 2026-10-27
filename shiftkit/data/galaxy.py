"""
Galaxy Zoo Evo images (SDSS -> DESI) from the SIDDA datasets on Zenodo.

Record: https://zenodo.org/records/14583107  (CC BY 4.0, doi:10.5281/zenodo.14583107)
Pandya, Patel, Nord, Walmsley & Ćiprijanović (2025), *Mach. Learn.: Sci. Technol.* 6(3), 035032.

The archive ``galaxy_dataset.zip`` is 2.4 GB and unpacks to about 19 GB of 100×100×3
float64 images (6 morphology classes).  Instead of downloading and unpacking it, this module
streams only the first ``n`` rows of each array with HTTP range requests and caches them as
small float32 files.  The arrays are shuffled (not sorted by class), so a prefix is a
representative sample; the class shares of the prefix are compared with the full label
arrays and a warning is issued if they differ by more than 3 percentage points.

Typical use
-----------
>>> data = load_galaxy_subset(cache_dir="./data/galaxy", n_train=3000, n_test=1000)
>>> data["x_train_sdss"].shape      # (3000, 3, 100, 100), float32, channels first
"""

import hashlib
import io
import os
import urllib.request
import warnings
import zipfile
from typing import Dict, Optional

import numpy as np

ZENODO_URL = "https://zenodo.org/records/14583107/files/galaxy_dataset.zip?download=1"
CLASSES = ["barred_spiral", "edge_on_disk", "featured_without_bar_or_spiral",
           "smooth_cigar", "smooth_round", "unbarred_spiral"]
_PREFIX = "galaxy_dataset"


class _RemoteFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests."""

    def __init__(self, url: str, chunk: int = 8 * 1024 * 1024):
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.url, self.size, self.pos, self.chunk = url, int(head.headers["Content-Length"]), 0, chunk
        self._buf_start, self._buf = 0, b""

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        end = self.pos + n
        if not (self._buf_start <= self.pos and end <= self._buf_start + len(self._buf)):
            stop = min(self.size, self.pos + max(n, self.chunk))
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{stop - 1}"})
            self._buf, self._buf_start = urllib.request.urlopen(req, timeout=300).read(), self.pos
        out = self._buf[self.pos - self._buf_start: end - self._buf_start]
        self.pos = end
        return out

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def _read_header(fh):
    version = np.lib.format.read_magic(fh)
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    return reader(fh)


def _first_rows(zf: zipfile.ZipFile, member: str, n: int) -> np.ndarray:
    """First *n* rows of an ``.npy`` member, read without extracting the member."""
    with zf.open(member) as fh:
        shape, fortran, dtype = _read_header(fh)
        if fortran:
            raise ValueError(f"{member} is Fortran-ordered; row prefix reading is not supported")
        n = min(n, shape[0])
        row = int(np.prod(shape[1:])) * dtype.itemsize
        raw = fh.read(n * row)
    return np.frombuffer(raw, dtype=dtype).reshape((n,) + tuple(shape[1:]))


def _cache_path(cache_dir: str, n_train: int, n_test: int) -> str:
    return os.path.join(cache_dir, f"galaxy_subset_train{n_train}_test{n_test}.npz")


def load_galaxy_subset(
    cache_dir: str = "./data/galaxy",
    n_train: int = 3000,
    n_test: int = 1000,
    url: str = ZENODO_URL,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Return a cached float32 subset of the SDSS/DESI galaxy images.

    Keys: ``x_train_sdss``, ``y_train_sdss``, ``x_train_desi``, ``y_train_desi``,
    ``x_test_sdss``, ``y_test_sdss``, ``x_test_desi``, ``y_test_desi``.
    Images are ``(n, 3, 100, 100)`` float32 in [0, 1]; labels are int64 in 0..5
    (see :data:`CLASSES`).  The first call downloads about
    ``(2·n_train + 2·n_test) × 0.03 MB`` of compressed data (about 240 MB for the default
    sizes, roughly 8 minutes at the 2–3 MB/s typically seen) and writes a cache of about
    ``(2·n_train + 2·n_test) × 0.12 MB`` (about 960 MB by default); later calls read the cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, n_train, n_test)
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f.files}

    if verbose:
        print(f"Downloading a {n_train}/{n_test} subset of the galaxy data from Zenodo "
              f"(streamed, first download only) ...", flush=True)
    try:
        zf = zipfile.ZipFile(_RemoteFile(url))
    except OSError as err:          # urllib.error.URLError / HTTPError are OSError subclasses
        raise RuntimeError(
            f"Could not reach {url} ({err}). Check the network connection and try again; "
            "nothing was written to the cache directory."
        ) from err
    out: Dict[str, np.ndarray] = {}
    for split, n in (("train", n_train), ("test", n_test)):
        for dom in ("sdss", "desi"):
            y_full = np.load(io.BytesIO(zf.read(f"{_PREFIX}/{split}/y_{split}_{dom}.npy")))
            y = y_full[:n].astype(np.int64)
            share_full = np.bincount(y_full, minlength=len(CLASSES)) / len(y_full)
            share = np.bincount(y, minlength=len(CLASSES)) / len(y)
            if np.abs(share - share_full).max() > 0.03:
                warnings.warn(f"{split}/{dom}: the first {n} rows differ from the full class shares "
                              f"by up to {np.abs(share - share_full).max():.3f}; the subset may not be representative.")
            x = _first_rows(zf, f"{_PREFIX}/{split}/x_{split}_{dom}.npy", n)
            out[f"x_{split}_{dom}"] = np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32)
            out[f"y_{split}_{dom}"] = y
            if verbose:
                print(f"  {split}/{dom}: {len(y)} images", flush=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)
    return out
