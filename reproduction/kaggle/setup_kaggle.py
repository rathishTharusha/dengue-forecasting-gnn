"""Verify the Kaggle CLI setup and fill kernel metadata with your username.

Deliberately never prints, copies, logs or transmits your credentials. It checks
only whether a credential exists and, for the legacy ``kaggle.json``, reads the
non-secret ``username`` field.

``kaggle >= 2.2`` replaced the old ``kaggle.json`` username/key pair with an
opaque access token, resolved in this order:

1. ``KAGGLE_API_TOKEN`` environment variable
2. ``~/.kaggle/access_token`` (or ``access_token.txt``)
3. ``~/.kaggle/kaggle.json`` (legacy)

``kaggle auth login`` does an OAuth flow and stores nothing you have to manage,
which is the option to prefer.

Usage::

    python reproduction/kaggle/setup_kaggle.py --check
    python reproduction/kaggle/setup_kaggle.py --init-metadata

Run ``--check`` first. It exits non-zero with an explanation if anything is
missing, so it is safe to use as a gate in a script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_DIR = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
KERNELS_DIR = Path(__file__).resolve().parent / "kernels"

# kaggle >= 2.2 resolves credentials in this order (kagglesdk.kaggle_env.
# get_access_token_from_env). The legacy kaggle.json is the last resort, and is
# what older docs describe -- including, wrongly, the first draft of this file.
LEGACY_JSON = KAGGLE_DIR / "kaggle.json"
TOKEN_FILES = (KAGGLE_DIR / "access_token", KAGGLE_DIR / "access_token.txt")


def _fail(message: str) -> int:
    print(f"FAIL  {message}")
    return 1


def check() -> int:
    """Confirm CLI, credentials and authentication, without exposing the key."""
    ok = True

    cli = shutil.which("kaggle")
    if cli:
        print(f"OK    kaggle CLI found at {cli}")
    else:
        ok = False
        print("FAIL  kaggle CLI not on PATH -- pip install kaggle")

    source = find_credentials()
    if source is None:
        print("FAIL  no Kaggle credentials found. Any one of these works:")
        print("        kaggle auth login                 (OAuth, nothing to store)")
        print("        set KAGGLE_API_TOKEN=<token>      (environment variable)")
        print(f"        {TOKEN_FILES[0]}   (token on one line)")
        print(f"        {LEGACY_JSON}      (legacy username/key pair)")
        print("      Generate a token at https://www.kaggle.com/settings/api")
        return 1

    print(f"OK    credentials found via {source}")

    # Permissions: a token is a bearer credential; keep it owner-only.
    for path in (*TOKEN_FILES, LEGACY_JSON):
        if path.exists() and os.name != "nt":
            mode = path.stat().st_mode & 0o777
            if mode != 0o600:
                print(f"WARN  {path} has mode {mode:o}; run: chmod 600 {path}")

    username = read_username()
    if username:
        print(f"OK    username: {username}")
    else:
        print("WARN  could not determine the username locally; --init-metadata will")
        print("      need it. Pass it explicitly: --init-metadata --username <name>")

    # Authenticate through the CLI so the key never passes through this process.
    probe = subprocess.run(
        ["kaggle", "kernels", "list", "--mine", "--page-size", "1"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        return _fail(f"API call failed: {detail[-1] if detail else 'unknown error'}")
    print("OK    API authenticated")

    return 0 if ok else 1


#: Byte-order marks that make a token file unreadable to the Kaggle SDK, which
#: calls plain ``Path.read_text()``. PowerShell 5.1's ``>`` redirect and
#: ``Out-File`` both write UTF-16LE with a BOM, so this is the single most likely
#: way a hand-created token file fails.
BOMS = (
    (bytes([0xFF, 0xFE]), "UTF-16LE with BOM (PowerShell 5.1 '>' or Out-File)"),
    (bytes([0xFE, 0xFF]), "UTF-16BE with BOM"),
    (bytes([0xEF, 0xBB, 0xBF]), "UTF-8 with BOM (Notepad)"),
)


def _bom_of(path: Path) -> str | None:
    """Return a description of the file's BOM, or None if it has none."""
    head = path.read_bytes()[:3]
    for marker, description in BOMS:
        if head.startswith(marker):
            return description
    return None


def find_credentials() -> str | None:
    """Name the credential source the CLI will use, or None. Never reads a token.

    Only inspects length and leading bytes -- never decodes or prints the value.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN environment variable"

    for path in TOKEN_FILES:
        if not path.exists():
            continue
        bom = _bom_of(path)
        if bom:
            print(f"FAIL  {path} is {bom}.")
            print("      The Kaggle SDK reads this file with a plain read_text() and")
            print("      will fail or read garbage. Rewrite it as plain UTF-8 with no")
            print("      BOM -- in cmd.exe (not PowerShell):")
            print(f'        > "{path}" echo YOUR_TOKEN')
            return None
        if path.read_bytes().strip():
            return str(path)

    if LEGACY_JSON.exists():
        return str(LEGACY_JSON)
    return None


def read_username() -> str | None:
    """Best-effort username lookup. Never returns, logs or transmits the token.

    An access token is opaque, so unlike the legacy ``kaggle.json`` there is no
    username to read out of it. Fall back to asking the API who we are.
    """
    if LEGACY_JSON.exists():
        try:
            data = json.loads(LEGACY_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if data.get("username"):
            return str(data["username"])

    # `kernels list --mine` echoes refs as "<username>/<slug>".
    probe = subprocess.run(
        ["kaggle", "kernels", "list", "--mine", "--page-size", "1", "--csv"],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        for line in probe.stdout.splitlines():
            if "/" in line:
                candidate = line.split(",")[0].strip()
                if "/" in candidate:
                    return candidate.split("/", 1)[0]
    return None


def init_metadata(username: str | None = None) -> int:
    """Substitute the username placeholder in every kernel-metadata.json."""
    username = username or read_username()
    if not username:
        return _fail(
            "could not determine your Kaggle username -- pass it with "
            "--username <name>, or run --check first"
        )

    if not KERNELS_DIR.exists():
        return _fail(f"no kernels directory at {KERNELS_DIR}")

    changed = 0
    for meta_path in sorted(KERNELS_DIR.glob("*/kernel-metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        slug = meta["id"].split("/", 1)[-1]
        new_id = f"{username}/{slug}"
        if meta["id"] == new_id:
            print(f"      unchanged  {meta_path.parent.name}  ({new_id})")
            continue
        meta["id"] = new_id
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"OK    set id     {meta_path.parent.name}  ({new_id})")
        changed += 1

    print(f"\n{changed} metadata file(s) updated.")
    print("Push one with:  cd reproduction/kaggle/kernels/<name> && kaggle kernels push")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify CLI, token and auth")
    parser.add_argument(
        "--init-metadata",
        action="store_true",
        help="write your username into every kernel-metadata.json",
    )
    parser.add_argument("--username", help="Kaggle username, if it cannot be detected")
    args = parser.parse_args()

    if not (args.check or args.init_metadata):
        parser.print_help()
        return 1

    status = 0
    if args.check:
        status |= check()
    if args.init_metadata:
        status |= init_metadata(args.username)
    return status


if __name__ == "__main__":
    sys.exit(main())
