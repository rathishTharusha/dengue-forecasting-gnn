"""Download output files and logs from Kaggle kernels safely with UTF-8 encoding.

Avoids Windows cp1252 charmap encoding errors in Kaggle CLI.

Usage::

    python reproduction/kaggle/download_kernel_outputs.py --kernels corrected-benchmark-stgat corrected-benchmark-a3tgcn --out-dir analysis/results/corrected_benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

REPO = Path(__file__).resolve().parent.parent.parent


def download_outputs(kernel_names: list[str], out_dir: Path, username: str | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    username = username or api.get_config_value("username") or "tharushaperera16"

    with api.build_kaggle_client() as kaggle:
        for kernel_name in kernel_names:
            slug = kernel_name.split("/")[-1]
            req = ApiListKernelSessionOutputRequest()
            req.user_name = username
            req.kernel_slug = slug
            try:
                res = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            except Exception as e:
                print(f"WARN  Failed to fetch outputs for {slug}: {e}")
                continue

            files_downloaded = 0
            for f in res.files or []:
                data = requests.get(f.url).content
                dest = out_dir / f.file_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                files_downloaded += 1
                print(f"  [OK] {slug} -> {f.file_name} ({len(data)} bytes)")

            if res.log:
                log_path = out_dir / f"{slug}.log"
                log_path.write_text(res.log, encoding="utf-8")
                print(f"  [OK] {slug} -> log saved ({len(res.log)} chars)")

            if files_downloaded == 0 and not res.log:
                print(f"  [--] {slug} output not ready yet.")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernels", nargs="+", required=True, help="Kernel names or slugs")
    ap.add_argument("--out-dir", default=str(REPO / "analysis" / "results" / "corrected_benchmark"))
    ap.add_argument("--username", default=None)
    args = ap.parse_args()

    return download_outputs(args.kernels, Path(args.out_dir), args.username)


if __name__ == "__main__":
    sys.exit(main())
