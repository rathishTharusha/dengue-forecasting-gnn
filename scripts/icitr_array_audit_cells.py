"""Notebook cells for the fast, local source-table and benchmark-array audit."""

from __future__ import annotations

from textwrap import dedent


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip()}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": dedent(source).strip(),
        "execution_count": None,
        "outputs": [],
    }


def array_audit_cells() -> list[dict]:
    """Return cells using the walkthrough's existing globals, without training.

    The source implementation is already embedded in MODULE_SOURCES. Every file
    this section creates is confined to a fresh workspace below RUN_DIR.
    """
    return [
        _markdown(
            """
            ### Recalculate the source-table and benchmark-array audit locally

            This section performs new calculations from the available source CSV,
            benchmark array and 25 daily ERA5 files. It rebuilds the district-week
            cases, dates the array by matching case vectors, checks whether future
            years enter training, and searches the weather alignment and lag sign.
            It trains no model and downloads nothing. The original implementation
            runs in a separate namespace; its plot and source inventory go to a
            fresh folder under `runs/icitr_notebook/`.

            **Two retained inputs remain explicit.** The correction's row A and
            row B come from `report_corrections.json`; the original PDF is not
            extracted here. For comparison with EXP-050, the array audit uses the
            historical dates in `full_paper/data/rebuilt_cases_weekly.csv`, including
            the eight dates later fixed by EXP-062. Those dates do not change the
            case-vector matches or imply that the old calendar should be used for
            new training. Missing local inputs are reported as unavailable.
            """
        ),
        _code(
            """
            import shutil
            import tempfile

            array_audit_assets = {
                "cases/output_Dengue Fever.csv": REPO / "full_paper/data/dengue_cases_raw.csv",
                "benchmark/sri_lanka_2013-2022_shifted.npy": REPO / "notebooks/baseline/sri_lanka_2013-2022_shifted.npy",
                "historical_cases_weekly.csv": REPO / "full_paper/data/rebuilt_cases_weekly.csv",
                "report_corrections.json": REPO / "data/external/report_corrections.json",
            }
            array_audit_assets.update({
                f"climate/{name}.json": REPO / "data/raw/era5_openmeteo" / f"{name}.json"
                for name in names
            })
            array_audit_missing = [str(path.relative_to(REPO)) for path in array_audit_assets.values()
                                   if not path.is_file()]
            array_audit_ready = not array_audit_missing
            array_audit_namespace = None
            if not array_audit_ready:
                display(Markdown("**Local source/array audit unavailable.** Missing inputs: "
                                 + ", ".join(f"`{path}`" for path in array_audit_missing)
                                 + ". No successful recalculation is claimed for this section."))
            else:
                array_audit_root = Path(tempfile.mkdtemp(prefix="array_audit_", dir=RUN_DIR)).resolve()
                array_audit_root.relative_to(RUN_DIR.resolve())
                array_audit_source = array_audit_root / "inputs"
                array_audit_output = array_audit_root / "outputs"
                array_audit_output.mkdir(parents=True)
                array_audit_inventory = []
                for relative, original in array_audit_assets.items():
                    target = array_audit_source / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, target)
                    array_audit_inventory.append({
                        "copied_as": relative,
                        "local_source": original.relative_to(REPO).as_posix(),
                        "bytes": target.stat().st_size,
                        "sha256_now": hashlib.sha256(target.read_bytes()).hexdigest(),
                    })
                graph_path = array_audit_source / "graph/sri_lanka_adj_list.json"
                graph_path.parent.mkdir()
                graph_path.write_text(json.dumps(HISTORICAL_GRAPH, indent=2), encoding="utf-8")
                array_audit_inventory.append({
                    "copied_as": "graph/sri_lanka_adj_list.json",
                    "local_source": "embedded historical graph from git 3878e70^",
                    "bytes": graph_path.stat().st_size,
                    "sha256_now": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                })
                pd.DataFrame(array_audit_inventory).to_csv(array_audit_root / "input_inventory.csv", index=False)

                array_audit_namespace = {
                    "__name__": "icitr_local_array_audit",
                    "SRC": array_audit_source, "OUT": array_audit_output,
                    "np": np, "pd": pd, "plt": plt, "json": json,
                    "ORIGINS": (0.55, 0.70, 0.85),
                    "rmse": lambda predicted, actual: float(np.sqrt(np.mean(
                        (np.asarray(predicted) - np.asarray(actual)) ** 2))),
                }
                data_source = MODULE_SOURCES["full_paper/kaggle/src/20_data.py"]
                # Reuse only the CSV parsing/mapping, ending before PDF extraction.
                csv_prefix = data_source.split("import pymupdf", 1)[0]
                exec(compile(csv_prefix, "embedded_case_table_prefix.py", "exec"), array_audit_namespace)
                correction = json.loads((array_audit_source / "report_corrections.json").read_text(encoding="utf-8"))[0]
                table_corrected = array_audit_namespace["table"].copy()
                correction_year = int(correction["report"]["year"])
                correction_number = int(correction["report"]["number"])
                for district, value in correction["replacement"].items():
                    selected = (table_corrected.year.eq(correction_year)
                                & table_corrected.week_no.eq(correction_number)
                                & table_corrected.district.eq(district))
                    table_corrected.loc[selected, "cases"] = value
                array_audit_namespace["tab"] = table_corrected
                # The original grid construction, without its date/PDF/weather code.
                grid_source = data_source[data_source.index('last_no = tab.groupby'):]
                grid_source = grid_source.split('dates = report_dates', 1)[0]
                exec(compile(grid_source, "embedded_week_grid.py", "exec"), array_audit_namespace)
                rebuilt_source_cases = array_audit_namespace["wide"].to_numpy(dtype=float)
                assert array_audit_namespace["NAMES"] == names
                check_close("source CSV plus retained correction reproduces corrected cases",
                            rebuilt_source_cases, cases,
                            "raw CSV -> original district mapping/grid + retained correction", atol=0)
                historical_case_table = pd.read_csv(array_audit_source / "historical_cases_weekly.csv",
                                                    parse_dates=["week_start"])
                check_close("historical date table carries the same weekly case counts",
                            historical_case_table[names].to_numpy(dtype=float), rebuilt_source_cases,
                            "historical date-table counts vs independently rebuilt source cases", atol=0)
                historical_dates = historical_case_table.week_start
                assert len(historical_dates) == len(rebuilt_source_cases) and historical_dates.notna().all()
                array_audit_namespace.update(
                    T=len(rebuilt_source_cases), CASES=rebuilt_source_cases,
                    MISSING=~array_audit_namespace["observed"], WEEK_START=historical_dates,
                )
                # Preserve the report's printed column order. These are retained
                # transcriptions, not newly extracted PDF contents.
                report_columns = ["Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "NuwaraEliya",
                                  "Galle", "Hambantota", "Matara", "Jaffna", "Kilinochchi", "Mannar",
                                  "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee",
                                  "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
                                  "Moneragala", "Ratnapura", "Kegalle", "Kalmunai", "SRILANKA"]
                array_audit_namespace["row_a"] = [correction["published_row_A"][name] for name in report_columns]
                array_audit_namespace["row_b"] = [correction["published_row_B"][name] for name in report_columns]
                print("New source-table reconstruction complete; original correction PDF not re-parsed.")
                print("Isolated audit folder:", array_audit_root)
            """
        ),
        _code(
            r"""
            if array_audit_ready:
                print("FRESH LOCAL ARRAY/WEATHER CALCULATION; retained correction rows and historical dates identified above")
                exec(compile(MODULE_SOURCES["full_paper/kaggle/src/30_audit.py"],
                             "embedded_benchmark_array_audit.py", "exec"), array_audit_namespace)
                audit = array_audit_namespace
                source_label = "local raw source table + benchmark array + daily ERA5 -> embedded 30_audit.py"
                check_close("array vectors matched to source reports", int(audit["matched"].sum()), 454,
                            source_label, atol=0)
                check_close("unmatched benchmark rows", np.flatnonzero(~audit["matched"]), [48, 49, 50, 54, 434],
                            source_label, atol=0)
                check_close("duplicated consecutive benchmark rows", audit["dups"], [444], source_label, atol=0)
                check_close("artifact benchmark row index", audit["k395"], 395, source_label, atol=0)
                check_close("artifact modeled-division total", audit["arr_cases"][audit["k395"]].sum(), 7165,
                            "benchmark array; raw CSV match identifies report", atol=0)
                check_close("artifact case vector matches retained printed row A",
                            audit["arr_cases"][audit["k395"]],
                            [correction["published_row_A"][name] for name in audit["NAMES"]],
                            "benchmark array vs retained printed-row transcription", atol=0)
                check_close("corrected modeled-division total", sum(audit["row_b"][:25]), 349,
                            "retained correction row B; not fresh PDF extraction", atol=0)
                formula_cells_now = sum(abs(audit["row_a"][k] - audit["row_a"][k-1] - audit["row_a"][k-2]) <= 1
                                        for k in range(2, 26))
                check_close("dragged-formula pattern cells", formula_cells_now, 15,
                            "new arithmetic on retained printed-row transcription", atol=0)
                leak_rows = []
                for origin in audit["ORIGINS"]:
                    cut = int(origin * len(audit["ids"]))
                    training_starts = audit["ids"][:cut-30]
                    future_rows = sum(audit["row_year"].iloc[row] == 2023 for row in training_starts)
                    leak_rows.append({"origin": origin, "2023 rows in training": future_rows})
                    check_close(f"benchmark origin {origin}: future-year rows in training", future_rows, 45,
                                source_label, atol=0)
                display(pd.DataFrame(leak_rows))
                check_close("rainfall lag-sign peak", audit["peak"], 12, source_label, atol=0)
                retained_log = (REPO / "full_paper/outputs/kaggle_run/notebook_output.txt").read_text(encoding="utf-8")
                temp_reference = re.search(r"array temperature vs ERA5:.*?r = ([0-9.]+).*?r = ([0-9.]+)", retained_log)
                rain_reference = re.search(r"best at k = \+12 \(r = ([0-9.]+)\); at the documented k = -12, r = ([0-9.]+)", retained_log)
                if temp_reference is None or rain_reference is None:
                    raise ValueError("Expected historical weather-audit comparison lines were not found.")
                # Historical correlations were printed to three decimal places.
                # The comparison therefore allows half a unit at that precision.
                weather_values = [audit["r_best"], audit["r_case"], max(audit["r_shift"]),
                                  audit["r_shift"][audit["shifts"].index(-12)]]
                weather_references = [float(temp_reference.group(1)), float(temp_reference.group(2)),
                                      float(rain_reference.group(1)), float(rain_reference.group(2))]
                weather_labels = ["temperature on its own timeline", "temperature on case-row dates",
                                  "rainfall twelve weeks later", "rainfall twelve weeks earlier"]
                weather_results = pd.DataFrame({"calculation": weather_labels,
                                                "fresh correlation": weather_values,
                                                "retained rounded correlation": weather_references})
                display(weather_results)
                for label, value, reference in zip(weather_labels, weather_values, weather_references):
                    check_close("weather audit: " + label, value, reference,
                                source_label + "; historical three-decimal log as comparison", atol=0.00050001)
                recorded_summary = json.loads((REPO / "full_paper/outputs/kaggle_run/results.json").read_text(encoding="utf-8"))
                check_close("benchmark array persistence floor", audit["ARRAY_FLOOR"],
                            [recorded_summary["array_floor"]["all_windows"],
                             recorded_summary["array_floor"]["without_row_395"]],
                            "new benchmark-array persistence predictions vs retained summary", atol=1e-8)
                weather_results.to_csv(array_audit_output / "weather_alignment.csv", index=False)
                pd.DataFrame(leak_rows).to_csv(array_audit_output / "future_rows_in_training.csv", index=False)
                print("Fresh benchmark persistence RMSE (all / excluding artifact-touching windows):", audit["ARRAY_FLOOR"])
                print("Fresh audit plot and CSVs:", array_audit_output)
            """
        ),
    ]
