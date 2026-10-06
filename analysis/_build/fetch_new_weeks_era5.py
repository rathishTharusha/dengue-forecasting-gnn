"""Fetch daily ERA5 for the new weeks (2024-01-01 to 2026-08-31) with the same Open-Meteo call as the
rebuilt series (``fetch_era5_climate.py``), into ``data/raw/era5_openmeteo_new`` (git-ignored).

    python analysis/_build/fetch_new_weeks_era5.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_era5_climate as era  # noqa: E402

era.RAW = era.REPO / "data" / "raw" / "era5_openmeteo_new"
era.START, era.END = "2024-01-01", "2026-08-31"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
era.fetch()
