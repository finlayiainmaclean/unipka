import hashlib
import numpy as np
import requests
from pathlib import Path
from platformdirs import user_cache_dir
from .version import __version__
import importlib.resources as res
from sklearn.linear_model import LogisticRegression, LinearRegression


ASSET_NAME = "t_dwar_v_novartis_a_b.pt"
ASSET_URL = f"https://github.com/finlayiainmaclean/unipka/releases/download/v{__version__}/{ASSET_NAME}"
ASSET_SHA256 = "48667090c330e745f0d91fd2eb159e04f41052d71b394e25d6ad2a2090d71c75"

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def get_model_path() -> Path:
    """Download and cache the asset, return its local path."""
    cache_dir = Path(user_cache_dir("unipka"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / ASSET_NAME

    if not file_path.exists():
        print(f"Downloading {ASSET_URL} ...")
        with requests.get(ASSET_URL, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    # Verify checksum
    if _sha256(file_path) != ASSET_SHA256:
        file_path.unlink(missing_ok=True)
        raise RuntimeError("Checksum mismatch for asset. Try again.")

    return file_path

def get_pattern_path(use_simple_smarts: bool = True):
    return res.files("unipka.data").joinpath("simple_smarts_pattern.tsv" if use_simple_smarts else "smarts_pattern.tsv")

def load_kpuu_model():
    weights_file = res.files("unipka.data").joinpath("weights.kpuu.npz")
    params = np.load(weights_file)
    clf = LogisticRegression()
    clf.fit(np.ones((2,3)), np.array([0,1])) # Dummy fit
    clf.coef_ = params["coef"]
    clf.intercept_ = params["intercept"]
    return clf






