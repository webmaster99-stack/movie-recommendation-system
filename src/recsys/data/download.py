"""Stage `download`: fetch MovieLens 32M from GroupLens, verify its SHA256, extract the files.

The checksum pinned in params.yaml guarantees everyone builds from byte-identical input.
The dataset's README.txt (which holds the license) is kept next to the CSVs, so it travels
with the data when pushed to the DVC remote, as the license requires.
"""

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

import httpx

from recsys.utils.config import load_params
from recsys.utils.paths import RAW_DIR

EXPECTED_FILES = ("ratings.csv", "movies.csv", "tags.csv", "links.csv", "README.txt")


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str | None) -> None:
    actual = sha256_of(path)
    if expected is None:
        raise ChecksumError(
            f"No checksum pinned for {path.name}. Verify the file, then set "
            f"data.movielens.sha256: {actual} in params.yaml."
        )
    if actual != expected:
        raise ChecksumError(f"{path.name}: expected sha256 {expected}, got {actual}")


def fetch(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)


def extract(zip_path: Path, out_dir: Path) -> None:
    """Extract the expected files flat into out_dir (the zip nests them under ml-32m/)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(name).name: name for name in zf.namelist() if not name.endswith("/")}
        missing = [f for f in EXPECTED_FILES if f not in members]
        if missing:
            raise FileNotFoundError(f"{zip_path.name} is missing {missing}")
        for name in EXPECTED_FILES:
            with zf.open(members[name]) as src, (out_dir / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)


def main() -> None:
    cfg = load_params()["data"]["movielens"]
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "ml-32m.zip"
        print(f"Downloading {cfg['url']} ...")
        fetch(cfg["url"], zip_path)
        verify_checksum(zip_path, cfg["sha256"])
        extract(zip_path, RAW_DIR)
    print(f"Extracted {len(EXPECTED_FILES)} files to {RAW_DIR}")


if __name__ == "__main__":
    main()
