"""Download Marvin's weights into weights/. Usage: python download_weights.py [79m] [23m] [--fp32]

For each model: the ONNX network and the small sub-second think-time head (.offset.npz). By default
the ONNX is the fp16-storage file: half the size, same outputs to within 0.001. Weights from before
the sub-second head are replaced.
"""

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "holymolyyy/marvin-onnx"
WEIGHTS = Path(__file__).resolve().parent / "weights"


def token() -> str | None:
    cached = Path.home() / ".cache" / "huggingface" / "token"
    return os.environ.get("HF_TOKEN") or (cached.read_text().strip() if cached.exists() else None)


def fetch(remote: str, dest: Path, auth: dict) -> None:
    url = f"https://huggingface.co/{REPO}/resolve/main/{remote}"
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=auth))
    except urllib.error.HTTPError as e:
        if e.code != 401 or not auth:
            raise
        response = urllib.request.urlopen(url)  # a stale token is rejected even for public files
    part = dest.with_name(dest.name + ".part")
    with response as r, open(part, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r{remote}: {done >> 20} / {total >> 20} MB", end="", flush=True)
    part.replace(dest)  # a new file: hard links to the old one keep the old contents
    print()


def main() -> None:
    WEIGHTS.mkdir(exist_ok=True)
    auth = {"Authorization": f"Bearer {t}"} if (t := token()) else {}
    args = [a for a in sys.argv[1:] if a != "--fp32"]
    suffix = "" if "--fp32" in sys.argv else "_fp16"
    for size in args or ["79m", "23m"]:
        onnx, offset = WEIGHTS / f"marvin_{size}.onnx", WEIGHTS / f"marvin_{size}.offset.npz"
        if onnx.exists() and offset.exists():
            print(f"{onnx.name}: already downloaded")
            continue
        # Without the offset head the network is from an older release: replace both.
        fetch(f"marvin_{size}{suffix}.onnx", onnx, auth)
        fetch(offset.name, offset, auth)


if __name__ == "__main__":
    main()
