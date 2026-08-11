"""Unit tests for libs.common.telemetry (plan §41) — no FastAPI involved."""
import math

import pytest

from libs.common.telemetry import (
    DEFAULT_BUCKETS_MS,
    Counter,
    Gauge,
    Histogram,
    Registry,
)

# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


def test_counter_increments_per_labelset():
    reg = Registry()
    c = reg.counter("reqs_total", "Requests.", ("method", "path"))
    c.inc(method="GET", path="/a")
    c.inc(method="GET", path="/a")
    c.inc(2.5, method="POST", path="/b")

    assert c.value(method="GET", path="/a") == 2.0
    assert c.value(method="POST", path="/b") == 2.5
    assert c.value(method="GET", path="/never") == 0.0


def test_counter_rejects_negative_and_bad_labels():
    reg = Registry()
    c = reg.counter("c_total", "C.", ("x",))
    with pytest.raises(ValueError):
        c.inc(-1.0, x="a")
    with pytest.raises(ValueError):
        c.inc(x="a", extra="nope")  # undeclared label
    with pytest.raises(ValueError):
        c.inc()  # missing declared label


def test_counter_without_labels_renders_bare_name():
    reg = Registry()
    c = reg.counter("bare_total", "Bare.")
    c.inc()
    c.inc(2)
    text = reg.render_prometheus()
    assert "# HELP bare_total Bare." in text
    assert "# TYPE bare_total counter" in text
    assert "\nbare_total 3\n" in text


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


def test_histogram_bucket_arithmetic_sum_and_count():
    reg = Registry()
    h = reg.histogram("lat_ms", "Latency.", ("path",))
    for v in (3.0, 7.0, 30.0, 6000.0):
        h.observe(v, path="/a")

    assert h.count(path="/a") == 4
    assert h.sum(path="/a") == pytest.approx(6040.0)

    cum = h.bucket_counts(path="/a")
    assert cum[5.0] == 1  # 3
    assert cum[10.0] == 2  # 3, 7
    assert cum[25.0] == 2
    assert cum[50.0] == 3  # + 30
    assert cum[5000.0] == 3  # 6000 is beyond the largest finite bucket
    assert cum[math.inf] == 4  # +Inf always equals count


def test_histogram_renders_cumulative_buckets_sum_count():
    reg = Registry()
    h = reg.histogram("lat_ms", "Latency.", ("path",))
    for v in (3.0, 7.0, 30.0, 6000.0):
        h.observe(v, path="/a")
    text = reg.render_prometheus()

    assert "# TYPE lat_ms histogram" in text
    assert 'lat_ms_bucket{path="/a",le="5"} 1\n' in text
    assert 'lat_ms_bucket{path="/a",le="10"} 2\n' in text
    assert 'lat_ms_bucket{path="/a",le="50"} 3\n' in text
    assert 'lat_ms_bucket{path="/a",le="5000"} 3\n' in text
    assert 'lat_ms_bucket{path="/a",le="+Inf"} 4\n' in text
    assert 'lat_ms_sum{path="/a"} 6040\n' in text
    assert 'lat_ms_count{path="/a"} 4\n' in text


def test_histogram_default_buckets_end_with_inf():
    reg = Registry()
    h = reg.histogram("h_ms", "H.")
    assert h.buckets == DEFAULT_BUCKETS_MS + (math.inf,)
    # boundary inclusion: an observation exactly at a bound lands IN it
    h.observe(5.0)
    assert h.bucket_counts()[5.0] == 1


# ---------------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------------


def test_gauge_set_and_callback():
    reg = Registry()
    g = reg.gauge("temperature", "Temp.")
    g.set(21.5)
    assert g.value() == 21.5

    ticks = iter([1.0, 2.0])
    g.set_callback(lambda: next(ticks))
    assert g.value() == 1.0  # callback evaluated on read
    text = reg.render_prometheus()
    assert "# TYPE temperature gauge" in text
    assert "\ntemperature 2\n" in text  # ... and again at render time


def test_gauge_labeled():
    reg = Registry()
    g = reg.gauge("age_days", "Age.", ("ticker",))
    g.set(3.0, ticker="AAPL")
    g.set(0.0, ticker="MSFT")
    text = reg.render_prometheus()
    assert 'age_days{ticker="AAPL"} 3\n' in text
    assert 'age_days{ticker="MSFT"} 0\n' in text


# ---------------------------------------------------------------------------
# Registry + exposition format
# ---------------------------------------------------------------------------


def test_render_emits_help_and_type_for_sampleless_metrics():
    reg = Registry()
    reg.counter("empty_total", "Never incremented.", ("x",))
    text = reg.render_prometheus()
    assert "# HELP empty_total Never incremented." in text
    assert "# TYPE empty_total counter" in text
    assert "empty_total{" not in text  # no samples, no sample lines
    assert text.endswith("\n")


def test_label_value_escaping():
    reg = Registry()
    c = reg.counter("esc_total", "Esc.", ("v",))
    # value contains a double quote, a literal backslash, and a newline
    c.inc(v='say "hi" \\ now\nplease')
    text = reg.render_prometheus()
    assert 'esc_total{v="say \\"hi\\" \\\\ now\\nplease"} 1\n' in text


def test_help_text_escaping():
    reg = Registry()
    reg.counter("help_total", "line one\nline two \\ backslash")
    text = reg.render_prometheus()
    assert "# HELP help_total line one\\nline two \\\\ backslash" in text


def test_registry_factories_are_idempotent_but_reject_conflicts():
    reg = Registry()
    a = reg.counter("dup_total", "Dup.", ("x",))
    b = reg.counter("dup_total", "Dup.", ("x",))
    assert a is b
    with pytest.raises(ValueError):
        reg.counter("dup_total", "Dup.", ("y",))  # different labelnames
    with pytest.raises(ValueError):
        reg.gauge("dup_total", "Dup.", ("x",))  # different type
