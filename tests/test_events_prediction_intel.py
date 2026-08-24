"""Pure prediction-market feature layer (Catalyst research upgrade, LOOP 4).

What these tests pin: the no-interpolation anchor rule (a delta with no
observation at/before its anchor is None — a young market has no honest
7-day change), the as-of filter (later observations never contaminate an
earlier view), the defined trend calculation, sample-size honesty
(observation_count/history_start/end), and the structural no-I/O guard.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from libs.trading_core.events import prediction_intel as pi

AS_OF = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Point:
    ts: datetime
    price: float


def hourly_series(prices: list[float], *, end: datetime = AS_OF) -> list[Point]:
    """Points ending AT `end`, spaced hourly, last price = prices[-1]."""
    return [
        Point(ts=end - timedelta(hours=len(prices) - 1 - i), price=p)
        for i, p in enumerate(prices)
    ]


# The no-I/O and no-numerics invariants for THIS module are enforced
# package-wide by tests/test_pure_layer_boundary.py, which walks every
# module under libs/trading_core/ — a per-file copy here protected this
# one file and left sixty-six others to the habit of copying a test.


def test_no_observations_is_none_never_a_zeroed_shell():
    assert pi.history_features([], as_of=AS_OF) is None
    future_only = [Point(ts=AS_OF + timedelta(hours=1), price=0.5)]
    assert pi.history_features(future_only, as_of=AS_OF) is None


def test_as_of_filter_excludes_later_observations():
    points = hourly_series([0.40, 0.45, 0.50]) + [
        Point(ts=AS_OF + timedelta(hours=2), price=0.99)
    ]
    features = pi.history_features(points, as_of=AS_OF)
    assert features.current_price == 0.50
    assert features.recent_high == 0.50  # the 0.99 is not knowable yet
    assert features.observation_count == 3
    assert features.history_end == AS_OF


def test_anchor_deltas_use_last_observation_at_or_before_anchor():
    # 26 hourly points reaching back 25h. The price AT the 1d anchor (0.45)
    # differs from its predecessor (0.40), so this pins the INCLUSIVE <=
    # rule: 0.15 under at-or-before, 0.20 under strictly-before.
    prices = [0.40, 0.45] + [0.40] * 22 + [0.55, 0.60]
    features = pi.history_features(hourly_series(prices), as_of=AS_OF)
    assert features.change_1h == 0.05   # 0.60 - 0.55
    assert features.change_1d == 0.15   # 0.60 - 0.45 (the point AT the anchor)
    # ...but the series is younger than 7 days: NO honest weekly change.
    assert features.change_7d is None


def test_future_anchors_yield_none_never_a_zero_self_delta():
    features = pi.history_features(
        hourly_series([0.40, 0.50]),
        as_of=AS_OF,
        previous_event_at=AS_OF + timedelta(days=1),
        window_start=AS_OF + timedelta(hours=1),
    )
    assert features.change_since_previous_event is None
    assert features.change_since_window_start is None


def test_same_instant_duplicates_collapse_to_the_last_listed_value():
    """The refetch-overwrites rule: a superseded value must vanish from
    EVERY statistic — count, high/low/range and the trend baseline."""
    points = [
        Point(ts=AS_OF - timedelta(hours=1), price=0.90),  # superseded
        Point(ts=AS_OF - timedelta(hours=1), price=0.40),  # the overwrite
        Point(ts=AS_OF, price=0.50),
    ]
    features = pi.history_features(points, as_of=AS_OF)
    assert features.observation_count == 2
    assert features.recent_high == 0.50  # 0.90 is gone everywhere
    assert features.trend == pi.TREND_RISING  # baseline is 0.40, not 0.90


def test_since_previous_event_and_window_start_anchors():
    prices = [0.30, 0.40, 0.50, 0.60]
    points = hourly_series(prices)
    features = pi.history_features(
        points,
        as_of=AS_OF,
        previous_event_at=AS_OF - timedelta(hours=2),
        window_start=AS_OF - timedelta(days=30),  # before the series began
    )
    assert features.change_since_previous_event == 0.20  # 0.60 - 0.40
    assert features.change_since_window_start is None    # no data there
    assert pi.history_features(points, as_of=AS_OF).change_since_previous_event is None


def test_high_low_range_and_defined_trend():
    rising = pi.history_features(
        hourly_series([0.40, 0.35, 0.55]), as_of=AS_OF
    )
    assert rising.recent_high == 0.55
    assert rising.recent_low == 0.35
    assert rising.price_range == 0.20
    assert rising.trend == pi.TREND_RISING  # +0.15 > threshold
    falling = pi.history_features(hourly_series([0.55, 0.40]), as_of=AS_OF)
    assert falling.trend == pi.TREND_FALLING
    flat = pi.history_features(hourly_series([0.50, 0.51]), as_of=AS_OF)
    assert flat.trend == pi.TREND_FLAT  # +0.01 inside the threshold
    # EXACTLY at the +/-0.02 boundary is FLAT ("beyond the threshold"), and
    # the comparison must survive float representation (0.42-0.40 != 0.02
    # in raw floats — the rounding rule makes it exact).
    at_boundary = pi.history_features(hourly_series([0.40, 0.42]), as_of=AS_OF)
    assert at_boundary.trend == pi.TREND_FLAT
    just_past = pi.history_features(hourly_series([0.40, 0.4201]), as_of=AS_OF)
    assert just_past.trend == pi.TREND_RISING


def test_naive_timestamps_treated_as_utc_and_deterministic():
    naive = [
        Point(ts=(AS_OF - timedelta(hours=1)).replace(tzinfo=None), price=0.4),
        Point(ts=AS_OF.replace(tzinfo=None), price=0.5),
    ]
    a = pi.history_features(naive, as_of=AS_OF.replace(tzinfo=None))
    b = pi.history_features(naive, as_of=AS_OF)
    assert a == b
    assert a.current_price == 0.5


def test_to_dict_carries_model_version():
    features = pi.history_features(hourly_series([0.4, 0.5]), as_of=AS_OF)
    payload = features.to_dict()
    assert payload["model_version"] == pi.PREDICTION_INTEL_MODEL_VERSION
    assert payload["history_start"].endswith("+00:00")


