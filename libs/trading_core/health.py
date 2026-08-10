"""Strategy Health Monitor v0 (development plan §19).

Pure functions over the chronological list of realized (closed-trade) PnLs.
No I/O, no clock reads unless the caller omits `as_of` — callers inject the
data and the timestamp, which keeps this module trivially testable and the
GET /api/health/strategy endpoint a thin wrapper.

Sign conventions (documented so the API response is unambiguous):
  - gross_profit_usd  >= 0 : sum of winning trades.
  - gross_loss_usd    >= 0 : MAGNITUDE of the summed losing trades.
  - avg_loss_usd (when defined) >= 0 : magnitude of the mean losing trade.
  - max_drawdown_usd / current_drawdown_usd >= 0 : magnitudes measured on the
    cumulative-PnL curve (starting equity 0).

Undefined statistics are None — never NaN/Infinity (they would not survive
JSON serialization): profit_factor is None when there are no losing trades,
win_rate/expectancy are None when there are no trades at all, avg_win/avg_loss
are None when there are no wins/losses respectively.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Status values for HealthStats.status (plan §19).
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
STATUS_HEALTHY = "HEALTHY"
STATUS_WARNING = "WARNING"
STATUS_PAUSE_RECOMMENDED = "PAUSE_RECOMMENDED"


@dataclass(frozen=True)
class HealthParams:
    """Monitoring parameters (plan §19; parameters, never hardcoded truths).

    - min_trades_for_judgement: below this many completed trades, judgement is
      withheld entirely (status INSUFFICIENT_DATA). §19: pause policies need a
      minimum sample of N completed trades — small-sample profit factors and
      win rates are statistical noise and must not trigger (or grant) a pause.
    - profit_factor_warning: profit factor below this -> WARNING.
    - profit_factor_pause: profit factor below this -> PAUSE_RECOMMENDED.
    - drawdown_warning_pct_of_gross: WARNING when the current drawdown exceeds
      this fraction of gross profit (drawdown material relative to what the
      strategy has actually earned).
    """

    min_trades_for_judgement: int = 10
    profit_factor_warning: float = 1.2
    profit_factor_pause: float = 1.0
    drawdown_warning_pct_of_gross: float = 0.5


@dataclass
class HealthStats:
    """Mirror of the GET /api/health/strategy response body (plan §19)."""

    as_of: str
    trade_count: int
    min_trades_for_judgement: int
    win_rate: float | None
    profit_factor: float | None
    expectancy_usd: float | None
    avg_win_usd: float | None
    avg_loss_usd: float | None
    gross_profit_usd: float
    gross_loss_usd: float
    cumulative_pnl_usd: float
    max_drawdown_usd: float
    current_drawdown_usd: float
    status: str
    explanations: list[str] = field(default_factory=list)


def compute_health(
    realized_pnls: list[float],
    params: HealthParams = HealthParams(),
    as_of: datetime | None = None,
) -> HealthStats:
    """Compute strategy health from chronological realized PnLs (plan §19).

    `as_of` is injectable for deterministic tests; it defaults to now (UTC)
    and only stamps the report — no statistic depends on it.
    """
    stamp = (as_of if as_of is not None else datetime.now(timezone.utc)).isoformat()

    n = len(realized_pnls)
    wins = [p for p in realized_pnls if p > 0]
    losses = [p for p in realized_pnls if p < 0]

    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))  # magnitude, >= 0

    win_rate = len(wins) / n if n > 0 else None
    expectancy = float(sum(realized_pnls)) / n if n > 0 else None
    avg_win = gross_profit / len(wins) if wins else None
    avg_loss = gross_loss / len(losses) if losses else None
    # No losses -> profit factor undefined: None, never +Infinity (plan §19,
    # and the API contract forbids NaN/Infinity in the JSON response).
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    # Drawdowns on the cumulative-PnL curve, starting from equity 0.
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in realized_pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    current_drawdown = peak - cumulative

    explanations: list[str] = []

    if n < params.min_trades_for_judgement:
        # §19: judgement withheld below the minimum sample — pause policies
        # need N completed trades; tiny samples are noise, so no status better
        # or worse than INSUFFICIENT_DATA is issued regardless of the numbers.
        status = STATUS_INSUFFICIENT_DATA
        explanations.append(
            f"Only {n} completed trades; judgement withheld below "
            f"min_trades_for_judgement={params.min_trades_for_judgement} (plan §19)."
        )
    else:
        explanations.append(
            f"{n} completed trades (>= min_trades_for_judgement="
            f"{params.min_trades_for_judgement}); judgement issued."
        )
        drawdown_threshold = params.drawdown_warning_pct_of_gross * gross_profit
        drawdown_breach = current_drawdown > drawdown_threshold

        if profit_factor is not None and profit_factor < params.profit_factor_pause:
            status = STATUS_PAUSE_RECOMMENDED
            explanations.append(
                f"Profit factor {profit_factor:.4f} < pause threshold "
                f"{params.profit_factor_pause} (gross profit {gross_profit:.2f} "
                f"vs gross loss {gross_loss:.2f}): the strategy is losing money."
            )
        elif (profit_factor is not None and profit_factor < params.profit_factor_warning) or drawdown_breach:
            status = STATUS_WARNING
            if profit_factor is not None and profit_factor < params.profit_factor_warning:
                explanations.append(
                    f"Profit factor {profit_factor:.4f} < warning threshold "
                    f"{params.profit_factor_warning}."
                )
            if drawdown_breach:
                explanations.append(
                    f"Current drawdown {current_drawdown:.2f} exceeds "
                    f"{params.drawdown_warning_pct_of_gross:.0%} of gross profit "
                    f"({drawdown_threshold:.2f})."
                )
        else:
            status = STATUS_HEALTHY
            if profit_factor is not None:
                explanations.append(
                    f"Profit factor {profit_factor:.4f} >= warning threshold "
                    f"{params.profit_factor_warning}."
                )
            else:
                explanations.append(
                    "No losing trades yet; profit factor undefined (reported as null)."
                )
            explanations.append(
                f"Current drawdown {current_drawdown:.2f} within "
                f"{params.drawdown_warning_pct_of_gross:.0%} of gross profit "
                f"({drawdown_threshold:.2f})."
            )

    return HealthStats(
        as_of=stamp,
        trade_count=n,
        min_trades_for_judgement=params.min_trades_for_judgement,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy_usd=expectancy,
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        gross_profit_usd=gross_profit,
        gross_loss_usd=gross_loss,
        cumulative_pnl_usd=float(sum(realized_pnls)),
        max_drawdown_usd=max_drawdown,
        current_drawdown_usd=current_drawdown,
        status=status,
        explanations=explanations,
    )
