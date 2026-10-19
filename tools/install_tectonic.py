"""Install the pinned official Tectonic binary for remote manuscript builds."""

import argparse
import hashlib
import io
from pathlib import Path
import tarfile
import urllib.request

VERSION = "0.17.0"
URL = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz"
SHA256 = "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(".tools/tectonic"))
    args = parser.parse_args()
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise RuntimeError("Official compiler archive checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        binaries = [entry for entry in archive.getmembers()
                    if entry.isfile() and Path(entry.name).name == "tectonic"]
        if len(binaries) != 1:
            raise RuntimeError("Compiler archive does not contain exactly one executable")
        data = archive.extractfile(binaries[0]).read()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes(data)
    args.destination.chmod(0o755)
    print(f"Installed Tectonic {VERSION} to {args.destination}; archive sha256 {SHA256}")


if __name__ == "__main__":
    main()
