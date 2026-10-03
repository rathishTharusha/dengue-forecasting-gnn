# %% [markdown]
# ## 9. Summary and outputs
#
# Everything below is written to `outputs/`: every run (`runs.json`, one row per
# configuration × origin × seed), the predictions (`predictions.pkl`), the tables as
# CSV, the figures, and `results.json` with the headline numbers. The paper's tables
# and figures are regenerated from these files.

# %%
LEV.to_csv(OUT / "levers.csv", index=False)
RESCUE.to_csv(OUT / "rescue.csv", index=False)
CMP.to_csv(OUT / "comparisons.csv", index=False)
ENC_PAIRS.to_csv(OUT / "encoder_heads.csv", index=False)
VT.to_csv(OUT / "validation_vs_test.csv", index=False)
DIAG.to_csv(OUT / "seir_decoder_diagnosis.csv")
RESID_CORR.to_csv(OUT / "residual_correlation.csv")
CEIL.to_csv(OUT / "error_structure.csv")
RECOVER.to_csv(OUT / "residual_recovery.csv")
RESULTS.update(profile=PROFILE, runs=int(len(done)), hours=round((time.time() - NOTEBOOK_START) / 3600, 2),
               credited_to_physics=credited)
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=float), encoding="utf-8")

b, p = RESULTS["persistence_3"], RESULTS["persistence_9"]
lev = LEV.set_index("lever")
cmp_ = CMP.set_index("comparison")
print("SUMMARY" + (" -- QUICK PROFILE, NOT RESULTS" if QUICK else ""))
print(f"  data: {T} weeks rebuilt from {len(SOURCES)} verified source files; lag-1 r2 {RESULTS['r2_lag1']:.2f}, "
      f"best climate r2 {RESULTS['r2_best_climate']:.2f}")
print(f"  benchmark array: persistence {ARRAY_FLOOR[0]:.2f} (all windows) / {ARRAY_FLOOR[1]:.2f} (without row 395); "
      f"2023 rows inside every training split")
print(f"  persistence (3 origins): val {b['val']:.2f}, test {b['test']:.2f}")
print(f"  best model B: val {mean_of(R3, 'B'):.2f}, test {mean_of(R3, 'B', 'RMSE'):.2f}")
print(f"  NB likelihood: {lev.loc['NB likelihood (vs squared error)', 'val delta']:+.2f} "
      f"({lev.loc['NB likelihood (vs squared error)', 'wins']})")
print(f"  SEIR repair credited to physics: {credited or 'none'}")
print(f"  gated SEIR-GNN vs SEIR-LSTM, 9 origins: val "
      f"{cmp_.loc['gated SEIR-GNN (AAGCN) vs SEIR-LSTM, 9 origins', 'val delta']:+.2f}, test "
      f"{cmp_.loc['gated SEIR-GNN (AAGCN) vs SEIR-LSTM, 9 origins', 'test delta']:+.2f}")
print(f"  finished in {RESULTS['hours']:.2f} h; outputs in {OUT}")
