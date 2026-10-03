from pathlib import Path
import sys
import numpy as np
import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis/_build"))
import run_covid_covariates as experiment


def test_external_inputs_exclude_unavailable_future():
    dates = pd.date_range("2020-01-01", periods=12, freq="7D")
    frame = pd.DataFrame(dict(week_start=dates, stringency_index=np.arange(12.),
                              mobility_workplaces=-np.arange(12.)))
    before, _ = experiment.external_inputs(frame, dates)
    changed = frame.copy()
    changed.loc[6:, ["stringency_index", "mobility_workplaces"]] = 999
    after, _ = experiment.external_inputs(changed, dates)
    np.testing.assert_array_equal(before[7], after[7])
    changed.loc[5, "stringency_index"] = 80
    after, _ = experiment.external_inputs(changed, dates)
    assert after[7, 0] == pytest.approx(.8)
    assert before[7, 0] == pytest.approx(.05)


def test_missing_inputs_have_neutral_gate():
    dates = pd.date_range("2020-01-01", periods=6, freq="7D")
    frame = pd.DataFrame(dict(week_start=dates, stringency_index=np.nan,
                              mobility_workplaces=np.nan))
    inputs, available = experiment.external_inputs(frame, dates)
    assert not available.any()
    assert (inputs == 0).all()
    assert frame.stringency_index.isna().all()


@pytest.mark.parametrize("arm", experiment.ARMS)
def test_zero_effect_matches_existing_model(arm):
    torch.set_num_threads(1)
    edge = torch.stack([torch.arange(25), torch.arange(25)])
    adj = torch.eye(25)
    torch.manual_seed(10)
    ref = experiment.OriginalModel("STGAT", 1, edge, adj, mod_clamp=.2)
    experiment.ACTIVE_ARM = arm
    torch.manual_seed(10)
    candidate = experiment.CovidModel("STGAT", 3, edge, adj, mod_clamp=.2)
    ref.eval()
    candidate.eval()
    x = torch.randn(2, 25, 3, 3)
    state = torch.zeros(2, 25, 4)
    state[..., 0] = .318
    state[..., 1:3] = .001
    state[..., 3] = .68
    with torch.no_grad():
        expected = ref(x[..., :1], state)
        actual = candidate(x, state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if arm != "base":
        candidate.covid_weight.data[:] = 1
        x[..., 1:] = 0
        with torch.no_grad():
            actual = candidate(x, state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
