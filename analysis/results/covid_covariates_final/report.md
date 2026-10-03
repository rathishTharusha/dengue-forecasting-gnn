# COVID policy and mobility: completed exploratory test

81 matched evaluations, nine origins and three seeds. Scores are means of split metrics, not globally pooled RMSE.

```text
            arm  val_RMSE  test_RMSE  test_MAE      bias  delta_vs_base  origins_won  origins_tied  p_vs_base  delta_ci_lower  delta_ci_upper  p_holm
           base 26.522965  30.048673 14.781874 -0.656655       0.000000            0             9    1.00000        0.000000        0.000000     NaN
         policy 26.627464  30.063735 14.820791 -0.918257       0.015062            3             3    0.90625       -0.156943        0.236408     1.0
policy_mobility 26.513977  30.069347 14.815282 -0.950614       0.020674            2             3    0.84375       -0.227763        0.231113     1.0
```

Origin-clustered exact sign flips; Holm adjustment for the two versus-base comparisons. Differences below 0.0001 RMSE count as numerical ties. Confidence intervals resample the nine origins, averaging seeds first; these nine-origin intervals have limited precision.

```text
                        test_RMSE   test_MAE       bias  pers_RMSE  train_policy_windows  train_mobility_windows  test_policy_windows  test_mobility_windows
arm             origin                                                                                                                                      
base            0.50    20.059440  11.171700  -2.593080  20.270668                   0.0                     0.0                  0.0                    0.0
                0.55    48.715675  23.278920 -14.777882  45.492813                   0.0                     0.0                  0.0                    0.0
                0.60    63.447854  32.286779  21.408719  48.714059                   0.0                     0.0                 19.0                   12.0
                0.65    11.535669   5.226099   0.089451  11.340509                   0.0                     0.0                 28.0                   28.0
                0.70    15.443915   6.309271  -0.643480  15.711850                  17.0                    10.0                 27.0                   27.0
                0.75    28.616966  11.299228  -1.520314  29.548785                  44.0                    37.0                 20.0                   20.0
                0.80    20.479815  12.403773  -1.273186  21.967747                  72.0                    65.0                 28.0                   28.0
                0.85    30.409112  14.678426  -4.226231  30.963665                  92.0                    85.0                 21.0                   13.0
                0.90    31.729609  16.382674  -2.373891  32.859071                 119.0                   112.0                  1.0                    0.0
policy          0.50    20.059444  11.171700  -2.593079  20.270668                   0.0                     0.0                  0.0                    0.0
                0.55    48.752127  23.297537 -14.811370  45.492813                   0.0                     0.0                  0.0                    0.0
                0.60    63.447854  32.286779  21.408719  48.714059                   0.0                     0.0                 19.0                   12.0
                0.65    11.535680   5.226108   0.089478  11.340509                   0.0                     0.0                 28.0                   28.0
                0.70    16.231732   6.552850  -0.891334  15.711850                  17.0                    10.0                 27.0                   27.0
                0.75    28.200486  11.362900  -2.110555  29.548785                  44.0                    37.0                 20.0                   20.0
                0.80    20.298132  12.432652  -2.790647  21.967747                  72.0                    65.0                 28.0                   28.0
                0.85    30.288127  14.576370  -5.085540  30.963665                  92.0                    85.0                 21.0                   13.0
                0.90    31.760032  16.480227  -1.479982  32.859071                 119.0                   112.0                  1.0                    0.0
policy_mobility 0.50    20.059444  11.171700  -2.593079  20.270668                   0.0                     0.0                  0.0                    0.0
                0.55    48.752127  23.297537 -14.811370  45.492813                   0.0                     0.0                  0.0                    0.0
                0.60    63.447854  32.286779  21.408719  48.714059                   0.0                     0.0                 19.0                   12.0
                0.65    11.535680   5.226108   0.089478  11.340509                   0.0                     0.0                 28.0                   28.0
                0.70    15.975094   6.478267  -1.054556  15.711850                  17.0                    10.0                 27.0                   27.0
                0.75    27.776072  11.173748  -2.132189  29.548785                  44.0                    37.0                 20.0                   20.0
                0.80    20.489131  12.416618  -2.176914  21.967747                  72.0                    65.0                 28.0                   28.0
                0.85    30.395629  14.684343  -4.974617  30.963665                  92.0                    85.0                 21.0                   13.0
                0.90    32.193092  16.602441  -2.310999  32.859071                 119.0                   112.0                  1.0                    0.0
```

Archived policy/mobility vintages are unverified. Latest external row is origin minus two weeks. Missing external data have neutral model effects. Training exposure counts do not establish variation or identification. No lockdown effect can be learned from an all-zero training exposure. This is an exploratory follow-up after previous searches, not independent confirmation or evidence of a causal policy effect.

The source-manifest boundary mismatch concerns an unused file. Checked tracked model inputs match HEAD (input_audit.json). The resume serialization fix changed the scheduler check only, with training definitions verified unchanged (resume_fix.json).

Hardware: origin
0.50     cpu
0.55     cpu
0.60     cpu
0.65     cpu
0.70     cpu
0.75     cpu
0.80     cpu
0.85     cpu
0.90    cuda
All arms and seeds within each origin use the same backend; CPU/CUDA training need not be bit-identical.
