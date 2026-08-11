"""In-process metrics + request-ID context (development plan §41).

Zero-dependency observability primitives shared by every service:

- :data:`request_id_var` — the contextvar binding one HTTP request's ID to
  everything that happens while serving it (logs, audit correlation_id,
  plan §38 + §41). Empty string when no request is in flight (scripts,
  startup).
- :class:`Counter` / :class:`Histogram` / :class:`Gauge` — minimal metric
  types with labels, aggregated in process memory.
- :class:`Registry` — creates metrics and renders them all in the
  Prometheus text exposition format 0.0.4, so ``GET /metrics`` is a plain
  string dump with no client library involved (plan §41: no external
  observability dependencies at V1).

House rule: stdlib only — this module must stay importable from any
context (gateway, scripts, tests) without FastAPI.
"""
from __future__ import annotations

import math
import threading
from contextvars import ContextVar
from typing import Callable

# The current HTTP request's ID (plan §41). Set by the gateway's request-ID
# middleware for the duration of one request; the empty-string default is
# the honest "no request context" value (scripts, tests, startup).
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Default latency buckets in MILLISECONDS (plan §41): request handling spans
# ~sub-10ms cache hits to multi-second backfills, hence the wide log-ish
# spread. +Inf is always appended.
DEFAULT_BUCKETS_MS: tuple[float, ...] = (
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
)

_INF = math.inf

# A labelset key: labels as a frozen tuple of (name, value) pairs in
# labelnames order — hashable, order-stable, and comparable across calls.
LabelKey = tuple[tuple[str, str], ...]


def _escape_help(text: str) -> str:
    r"""Escape a HELP line per exposition format 0.0.4 (``\`` and newline)."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label_value(value: str) -> str:
    r"""Escape a label value per 0.0.4 (``\``, ``"`` and newline)."""
    return (
        value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    )


def _format_number(value: float) -> str:
    """Render a sample value: ``+Inf`` for infinity, integers without ``.0``."""
    if value == _INF:
        return "+Inf"
    if value == -_INF:
        return "-Inf"
    if value == int(value):
        return str(int(value))
    return repr(value)


def _render_labels(key: LabelKey, extra: tuple[tuple[str, str], ...] = ()) -> str:
    """Render ``{a="x",b="y"}`` (empty string when there are no labels)."""
    pairs = key + extra
    if not pairs:
        return ""
    inner = ",".join(f'{name}="{_escape_label_value(value)}"' for name, value in pairs)
    return "{" + inner + "}"


class _Metric:
    """Shared labelset bookkeeping for all three metric types."""

    kind = "untyped"

    def __init__(self, name: str, help: str, labelnames: tuple[str, ...]) -> None:
        self.name = name
        self.help = help
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()

    def _key(self, labels: dict[str, str]) -> LabelKey:
        """Validate + freeze one call's labels into a :data:`LabelKey`.

        Every declared label must be provided (no partial labelsets) and no
        undeclared label may sneak in — silent typos would fork time series.
        """
        if set(labels) != set(self.labelnames):
            raise ValueError(
                f"{self.name}: labels {sorted(labels)} do not match declared "
                f"labelnames {sorted(self.labelnames)}"
            )
        return tuple((name, str(labels[name])) for name in self.labelnames)

    def _header(self) -> list[str]:
        return [
            f"# HELP {self.name} {_escape_help(self.help)}",
            f"# TYPE {self.name} {self.kind}",
        ]

    def render(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError


class Counter(_Metric):
    """Monotonically increasing counter with labels (plan §41)."""

    kind = "counter"

    def __init__(self, name: str, help: str, labelnames: tuple[str, ...] = ()) -> None:
        super().__init__(name, help, labelnames)
        self._values: dict[LabelKey, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Add ``amount`` (>= 0 — counters never go down) to one labelset."""
        if amount < 0:
            raise ValueError(f"{self.name}: counter increment must be >= 0, got {amount}")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        """Current value for one labelset (0.0 when never incremented)."""
        return self._values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        lines = self._header()
        with self._lock:
            for key, value in self._values.items():
                lines.append(
                    f"{self.name}{_render_labels(key)} {_format_number(value)}"
                )
        return lines


class Histogram(_Metric):
    """Fixed-bucket latency histogram (plan §41).

    Buckets are upper bounds in MILLISECONDS (:data:`DEFAULT_BUCKETS_MS` by
    default) with ``+Inf`` always appended; per labelset it tracks bucket
    counts plus the running sum and count, exactly what the Prometheus
    ``_bucket`` / ``_sum`` / ``_count`` exposition needs.
    """

    kind = "histogram"

    def __init__(
        self,
        name: str,
        help: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_BUCKETS_MS,
    ) -> None:
        super().__init__(name, help, labelnames)
        bounds = tuple(sorted(float(b) for b in buckets))
        if not bounds:
            raise ValueError(f"{self.name}: at least one bucket bound required")
        if bounds[-1] != _INF:
            bounds = bounds + (_INF,)
        self.buckets = bounds
        # Per labelset: ([non-cumulative bucket counts], sum, count).
        self._series: dict[LabelKey, list] = {}

    def observe(self, value: float, **labels: str) -> None:
        """Record one observation into its bucket + sum/count."""
        key = self._key(labels)
        with self._lock:
            series = self._series.get(key)
            if series is None:
                series = [[0] * len(self.buckets), 0.0, 0]
                self._series[key] = series
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    series[0][i] += 1
                    break
            series[1] += value
            series[2] += 1

    def count(self, **labels: str) -> int:
        """Total observations for one labelset (0 when never observed)."""
        series = self._series.get(self._key(labels))
        return series[2] if series else 0

    def sum(self, **labels: str) -> float:
        """Sum of observations for one labelset (0.0 when never observed)."""
        series = self._series.get(self._key(labels))
        return series[1] if series else 0.0

    def bucket_counts(self, **labels: str) -> dict[float, int]:
        """CUMULATIVE count per bucket bound, as exposed to Prometheus."""
        series = self._series.get(self._key(labels))
        counts = series[0] if series else [0] * len(self.buckets)
        out: dict[float, int] = {}
        running = 0
        for bound, n in zip(self.buckets, counts):
            running += n
            out[bound] = running
        return out

    def render(self) -> list[str]:
        lines = self._header()
        with self._lock:
            for key, (counts, total, count) in self._series.items():
                running = 0
                for bound, n in zip(self.buckets, counts):
                    running += n
                    le = (("le", _format_number(bound)),)
                    lines.append(
                        f"{self.name}_bucket{_render_labels(key, le)} {running}"
                    )
                lines.append(
                    f"{self.name}_sum{_render_labels(key)} {_format_number(total)}"
                )
                lines.append(f"{self.name}_count{_render_labels(key)} {count}")
        return lines


class Gauge(_Metric):
    """Point-in-time value, either set directly or via callback (plan §41).

    ``set()`` stores a value; ``set_callback()`` stores a zero-arg callable
    evaluated at scrape time (e.g. process uptime). A callback wins over a
    previously set value for the same labelset and vice versa — last call
    decides.
    """

    kind = "gauge"

    def __init__(self, name: str, help: str, labelnames: tuple[str, ...] = ()) -> None:
        super().__init__(name, help, labelnames)
        self._values: dict[LabelKey, float | Callable[[], float]] = {}

    def set(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def set_callback(self, fn: Callable[[], float], **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = fn

    def value(self, **labels: str) -> float:
        """Current value for one labelset (callbacks are evaluated)."""
        stored = self._values[self._key(labels)]
        return float(stored()) if callable(stored) else stored

    def render(self) -> list[str]:
        lines = self._header()
        with self._lock:
            items = list(self._values.items())
        for key, stored in items:
            value = float(stored()) if callable(stored) else stored
            lines.append(f"{self.name}{_render_labels(key)} {_format_number(value)}")
        return lines


class Registry:
    """Creates metrics and renders them all (plan §41).

    The factories are idempotent: asking again for an existing name with the
    same type and labelnames returns the SAME metric object (safe on module
    re-import), while a conflicting re-registration raises — two definitions
    of one name would silently split its samples.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _register(self, cls: type, name: str, help: str, labelnames: tuple[str, ...], **kwargs):
        with self._lock:
            existing = self._metrics.get(name)
            if existing is not None:
                if type(existing) is not cls or existing.labelnames != tuple(labelnames):
                    raise ValueError(
                        f"metric {name!r} already registered with a different "
                        "type or labelnames"
                    )
                return existing
            metric = cls(name, help, tuple(labelnames), **kwargs)
            self._metrics[name] = metric
            return metric

    def counter(self, name: str, help: str, labelnames: tuple[str, ...] = ()) -> Counter:
        return self._register(Counter, name, help, labelnames)

    def histogram(
        self,
        name: str,
        help: str,
        labelnames: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_BUCKETS_MS,
    ) -> Histogram:
        return self._register(Histogram, name, help, labelnames, buckets=buckets)

    def gauge(self, name: str, help: str, labelnames: tuple[str, ...] = ()) -> Gauge:
        return self._register(Gauge, name, help, labelnames)

    def render_prometheus(self) -> str:
        """All metrics in Prometheus text exposition format 0.0.4."""
        lines: list[str] = []
        with self._lock:
            metrics = list(self._metrics.values())
        for metric in metrics:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"


# The process-wide default registry every service registers into; the
# gateway's GET /metrics renders exactly this (plan §41).
REGISTRY = Registry()
