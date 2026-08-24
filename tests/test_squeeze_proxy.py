"""Squeeze-proxy lib tests (auto-strategy Phase D). The proxy must:
compute a hand-checkable volume z, fire on crowded-breakout and gap-up,
stay silent when unmeasurable (a blind proxy must not cry wolf), and
always state its reasons."""
from libs.trading_core.risk.squeeze import SqueezeProxyParams, assess_squeeze_proxy


def flat_series(n, close=100.0, vol=1_000_000.0):
    return (
        [close] * n,          # opens
        [close] * n,          # closes
        [close + 0.5] * n,    # highs
        [vol] * n,            # volumes
    )


def test_volume_z_hand_computed_and_crowded_breakout():
    opens, closes, highs, volumes = flat_series(40)
    # prior 20 volumes alternate 0.9M/1.1M -> mean 1.0M, stdev ~0.1026M
    for i in range(-21, -1):
        volumes[i] = 900_000.0 if i % 2 else 1_100_000.0
    volumes[-1] = 1_500_000.0  # ~4.9 sigma
    closes[-1] = 100.4  # within 5% of the 100.5 trailing high
    r = assess_squeeze_proxy(opens, closes, highs, volumes)
    assert r.volume_z is not None and 4.0 < r.volume_z < 6.0
    assert r.dist_from_high_pct is not None and r.dist_from_high_pct < 5.0
    assert r.elevated
    assert any("volume z" in x for x in r.reasons)


def test_gap_up_alone_fires():
    opens, closes, highs, volumes = flat_series(40)
    opens[-1] = 106.0  # +6% overnight vs prior close 100
    r = assess_squeeze_proxy(opens, closes, highs, volumes)
    assert r.overnight_gap_pct is not None and r.overnight_gap_pct > 5.0
    assert r.elevated
    assert any("gap-up" in x for x in r.reasons)


def test_high_volume_far_from_high_does_not_fire():
    opens, closes, highs, volumes = flat_series(40)
    for i in range(-21, -1):
        volumes[i] = 900_000.0 if i % 2 else 1_100_000.0
    volumes[-1] = 2_000_000.0
    closes[-1] = 80.0  # 20% off the trailing high
    highs[-1] = 80.5
    r = assess_squeeze_proxy(opens, closes, highs, volumes)
    assert r.elevated is False


def test_unmeasurable_is_silent_with_reasons():
    r = assess_squeeze_proxy([100.0], [100.0], [100.5], [1e6])
    assert r.volume_z is None and r.overnight_gap_pct is None
    assert r.elevated is False
    assert r.reasons  # says WHY it cannot measure


def test_zero_stdev_volume_is_honest_null():
    opens, closes, highs, volumes = flat_series(40)  # constant volume
    r = assess_squeeze_proxy(opens, closes, highs, volumes)
    assert r.volume_z is None
    assert any("stdev" in x for x in r.reasons)
