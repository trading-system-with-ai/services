"""Risk-informed portfolio-backtest advice (user mandate 2026-08-20:
"回测时结合风控算法模型,对组合进行建议,并说明理由").

Every item is DETERMINISTIC QUANT OUTPUT computed by the SAME risk-model
libraries the live platform runs (§21): historical VaR/ES from
risk/models/var_es.py (method-labelled ModelResult, never a bare number,
§6), drawdown from risk/models/drawdown.py, Spearman correlation from
correlation.py. No LLM, no invented thresholds presented as truths —
every trigger level is a parameter (§6.2) and every item carries its
evidence numbers and its rationale. Server strings are English verbatim
(§26/§36 convention: analysis text is an audit-worthy exact record).

Honest empties: too little history → the item SAYS so instead of
guessing; an empty list means "nothing rose to advice", not "no risk".

Severity rule (uniform): WARNING = a breach REALIZED in this replay
(drawdown, wipe-out); SUGGESTION = an estimated / forward-looking breach
(tail, concentration, correlation); INFO = context indicating no change.
Trigger levels are parameters; each fired item echoes its warn level in
``evidence`` so the stored record is self-describing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from libs.trading_core.correlation import spearman
from libs.trading_core.risk.models.base import ModelHealth
from libs.trading_core.risk.models.drawdown import drawdown
from libs.trading_core.risk.models.var_es import historical_es, historical_var


@dataclass(frozen=True)
class AdviceParams:
    """Research trigger levels (§6.2 — parameters, never truths)."""

    var_confidence: float = 0.95
    es_confidence: float = 0.975
    var_warn_pct_of_equity: float = 0.03  # 1-day VaR > 3% of equity
    max_drawdown_warn: float = -0.20
    concentration_warn_pct: float = 30.0  # single-name |allocation| percent
    correlation_warn: float = 0.7
    cash_drag_days_pct: float = 0.6  # >60% of bars ≥99% cash


@dataclass(frozen=True)
class AdviceItem:
    """Text fields are BILINGUAL pairs {"en": ..., "zh": ...} — both are
    generated from the same deterministic template in the same call (user
    mandate 2026-08-20: advice must be readable in Chinese), so the two
    languages cannot drift. Evidence stays language-neutral data."""

    severity: str  # INFO | SUGGESTION | WARNING
    code: str
    finding: dict
    evidence: dict
    suggestion: dict
    rationale: dict


def assess_portfolio_result(
    dates: list[date],
    equity: list[float],
    allocations: list[dict[str, float]],
    cash_pct: list[float],
    symbol_closes: dict[str, list[float]],
    params: AdviceParams = AdviceParams(),
) -> list[AdviceItem]:
    items: list[AdviceItem] = []
    n = len(equity)
    if n < 2:
        return [AdviceItem(
            severity="INFO", code="INSUFFICIENT_DATA",
            finding={"en": f"only {n} bars — no assessment is honest at this length",
                     "zh": f"仅 {n} 根K线 — 该长度下任何评估都不诚实"},
            evidence={"bars": n},
            suggestion={"en": "run over a longer shared window",
                        "zh": "请在更长的共同时间窗上运行"},
            rationale={"en": "every model below needs a return series; none exists",
                       "zh": "以下所有模型都需要收益序列,而当前没有"},
        )]

    # ---- tail risk: the SAME historical VaR/ES models the live platform
    # runs, method-labelled (§6: no VaR without its method). Run on the
    # RETURN series (scale-free — verifier catch: a USD tail divided by
    # FINAL equity misstates risk on any growing or shrinking book); the
    # USD figure at final equity is secondary evidence, labelled as such.
    returns = [
        equity[t] / equity[t - 1] - 1.0
        for t in range(1, n)
        if equity[t - 1] > 0
    ]
    final_equity = equity[-1]
    var_res = historical_var(returns, params.var_confidence, as_of=dates[-1])
    es_res = historical_es(returns, params.es_confidence, as_of=dates[-1])
    if var_res.value is not None:
        # ACTIVE or DEGRADED both carry a real number; DEGRADED states why
        # in the evidence instead of being hidden (§44 rule 18).
        var_pct = var_res.value  # a FRACTION of then-current equity, natively
        evidence = {
            "var_pct_of_equity": round(var_pct * 100.0, 2),
            "var_usd_at_final_equity": (
                round(var_pct * final_equity, 2) if final_equity > 0 else None
            ),
            "es_pct_of_equity": (
                round(es_res.value * 100.0, 2) if es_res.value is not None else None
            ),
            "model_health": var_res.health.value,
            "health_reason": var_res.reason,
            "confidence": params.var_confidence,
            "warn_level_pct": params.var_warn_pct_of_equity * 100.0,
            "method": "HISTORICAL (empirical, k-th largest loss, on daily returns)",
            "sample_days": var_res.sample_size,
        }
        if var_pct > params.var_warn_pct_of_equity:
            items.append(AdviceItem(
                severity="SUGGESTION", code="TAIL_RISK",
                finding={
                    "en": f"1-day VaR{int(params.var_confidence * 100)} was {var_pct:.1%} of equity",
                    "zh": f"单日 VaR{int(params.var_confidence * 100)} 达净值的 {var_pct:.1%}",
                },
                evidence=evidence,
                suggestion={
                    "en": (
                        "lower per-entry risk (smaller tier budgets via risk "
                        "limits) or max_gross_pct so a single bad day stays "
                        f"under {params.var_warn_pct_of_equity:.0%} of equity"
                    ),
                    "zh": (
                        "降低单笔风险(调小风险限额中的分档预算)或调低 "
                        f"max_gross_pct,使单个坏日的亏损保持在净值的 "
                        f"{params.var_warn_pct_of_equity:.0%} 以内"
                    ),
                },
                rationale={
                    "en": (
                        "the historical daily RETURN distribution of THIS replay "
                        "— same estimator the live risk engine runs — implies a "
                        "1-in-20 day loses more than the stated fraction of the "
                        "then-current book"
                    ),
                    "zh": (
                        "本次回放的历史日收益分布(与实盘风险引擎同一估计器)"
                        "意味着每 20 天约有 1 天的亏损超过上述占当时净值的比例"
                    ),
                },
            ))
        else:
            items.append(AdviceItem(
                severity="INFO", code="TAIL_RISK",
                finding={
                    "en": f"1-day VaR{int(params.var_confidence * 100)} within tolerance",
                    "zh": f"单日 VaR{int(params.var_confidence * 100)} 在容忍范围内",
                },
                evidence=evidence,
                suggestion={"en": "no change indicated by the tail",
                            "zh": "尾部风险未提示需要任何调整"},
                rationale={"en": "empirical daily-loss tail below the warn level",
                           "zh": "经验日亏损尾部低于警戒水平"},
            ))
    else:
        items.append(AdviceItem(
            severity="INFO", code="TAIL_RISK",
            finding={"en": "VaR/ES unavailable for this replay",
                     "zh": "本次回放无法计算 VaR/ES"},
            evidence={"health": var_res.health.value, "reason": var_res.reason},
            suggestion={"en": "run over a longer window for a valid tail estimate",
                        "zh": "请在更长时间窗上运行以获得有效的尾部估计"},
            rationale={"en": "the model refuses rather than guesses below min_obs (§44 rule 18)",
                       "zh": "样本不足时模型选择拒绝而非猜测(§44 规则 18)"},
        ))

    # ---- drawdown: the live §2.8 model over the replay NAV path.
    if min(equity) <= 0.0:
        # A wiped-out book has no percentage drawdown — say so LOUDLY
        # (verifier catch: silence here read as "no drawdown finding").
        items.append(AdviceItem(
            severity="WARNING", code="DRAWDOWN",
            finding={"en": "equity reached zero or below — drawdown has no percentage interpretation",
                     "zh": "净值触及零或以下 — 回撤失去百分比意义"},
            evidence={
                "min_equity": round(min(equity), 2),
                "final_equity": round(final_equity, 2),
            },
            suggestion={"en": "the book was wiped out — re-run with lower max_gross_pct",
                        "zh": "账户已被击穿 — 请调低 max_gross_pct 后重新运行"},
            rationale={"en": "a drawdown is a ratio to a peak; a NAV <= 0 breaks the ratio",
                       "zh": "回撤是相对峰值的比率;净值 ≤ 0 使该比率无法定义"},
        ))
        dd = None
    else:
        dd = drawdown(list(zip(dates, equity)))
    if dd is not None:
        if dd.is_available and dd.max_dd_pct is not None:
            evidence = {
                "max_drawdown_pct": round(dd.max_dd_pct * 100.0, 2),
                "peak_date": dd.peak_date.isoformat() if dd.peak_date else None,
                "trough_date": dd.trough_date.isoformat() if dd.trough_date else None,
                "warn_level_pct": params.max_drawdown_warn * 100.0,
            }
            if dd.max_dd_pct <= params.max_drawdown_warn:
                items.append(AdviceItem(
                    severity="WARNING", code="DRAWDOWN",
                    finding={"en": f"max drawdown {dd.max_dd_pct:.1%}",
                             "zh": f"最大回撤 {dd.max_dd_pct:.1%}"},
                    evidence=evidence,
                    suggestion={
                        "en": (
                            "tighten max_gross_pct, raise cash_floor_pct, or cap "
                            "max_positions — all are portfolio request parameters"
                        ),
                        "zh": (
                            "收紧 max_gross_pct、提高 cash_floor_pct 或限制 "
                            "max_positions — 三者都是组合回测的请求参数"
                        ),
                    },
                    rationale={
                        "en": (
                            "the equity path itself breached the drawdown warn "
                            "level; the allocation table shows what was held "
                            "between peak and trough"
                        ),
                        "zh": (
                            "净值路径实际突破了回撤警戒线;分配表可以看到峰谷"
                            "之间持有了什么"
                        ),
                    },
                ))
            else:
                items.append(AdviceItem(
                    severity="INFO", code="DRAWDOWN",
                    finding={"en": f"max drawdown {dd.max_dd_pct:.1%} within tolerance",
                             "zh": f"最大回撤 {dd.max_dd_pct:.1%},在容忍范围内"},
                    evidence=evidence,
                    suggestion={"en": "no change indicated by the drawdown",
                                "zh": "回撤未提示需要任何调整"},
                    rationale={"en": "worst peak-to-trough stayed above the warn level",
                               "zh": "最差峰谷回撤未达警戒水平"},
                ))

    # ---- concentration: peak single-name |allocation|.
    peak_pct = 0.0
    peak_signed = 0.0
    peak_day: date | None = None
    peak_ticker = ""
    for t, row in enumerate(allocations):
        for tk, v in row.items():
            if abs(v) > peak_pct:
                peak_pct, peak_signed, peak_day, peak_ticker = abs(v), v, dates[t], tk
    if peak_day is not None and peak_pct >= params.concentration_warn_pct:
        side = "SHORT" if peak_signed < 0 else "LONG"
        items.append(AdviceItem(
            severity="SUGGESTION", code="CONCENTRATION",
            finding={
                "en": (
                    f"single-name exposure peaked at {peak_signed:+.1f}% "
                    f"({side} {peak_ticker}, {peak_day.isoformat()})"
                ),
                "zh": (
                    f"单一标的敞口峰值 {peak_signed:+.1f}%"
                    f"({'空头' if side == 'SHORT' else '多头'} {peak_ticker},"
                    f"{peak_day.isoformat()})"
                ),
            },
            evidence={"ticker": peak_ticker, "date": peak_day.isoformat(),
                      "peak_abs_allocation_pct": round(peak_pct, 1),
                      "peak_signed_allocation_pct": round(peak_signed, 1),
                      "side": side,
                      "warn_level_pct": params.concentration_warn_pct},
            suggestion={
                "en": (
                    "lower position_pct or add symbols so no single name "
                    f"carries more than ~{params.concentration_warn_pct:.0f}% of equity"
                ),
                "zh": (
                    f"调低 position_pct 或增加标的,使任何单一标的的敞口"
                    f"不超过净值的约 {params.concentration_warn_pct:.0f}%"
                ),
            },
            rationale={
                "en": (
                    "one name dominating the book makes portfolio P&L a "
                    "single-stock bet; the daily allocation table shows the day"
                ),
                "zh": (
                    "单一标的主导组合会让组合盈亏变成单股赌注;"
                    "分配表可定位到具体日期"
                ),
            },
        ))

    # ---- correlation: pairwise Spearman of daily returns (live lib).
    def pair_returns(ca: list[float], cb: list[float]) -> tuple[list[float], list[float]]:
        """Date-ALIGNED return pairs: a bar unusable for either leg drops
        from BOTH (verifier catch: per-series filtering misaligned the
        samples and spearman raises on unequal lengths)."""
        ra: list[float] = []
        rb: list[float] = []
        for t in range(1, min(len(ca), len(cb))):
            if ca[t - 1] > 0 and cb[t - 1] > 0:
                ra.append(ca[t] / ca[t - 1] - 1.0)
                rb.append(cb[t] / cb[t - 1] - 1.0)
        return ra, rb

    tickers = sorted(symbol_closes)
    correlated: list[tuple[str, str, float]] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            ra, rb = pair_returns(symbol_closes[a], symbol_closes[b])
            if len(ra) < 2:
                continue  # too little shared history — no honest rho
            rho = spearman(ra, rb)
            if rho is not None and abs(rho) >= params.correlation_warn:
                correlated.append((a, b, rho))
    if correlated:
        items.append(AdviceItem(
            severity="SUGGESTION", code="CORRELATION",
            finding={
                "en": (
                    "highly correlated pairs: "
                    + ", ".join(f"{a}/{b} ρ={r:.2f}" for a, b, r in correlated)
                ),
                "zh": (
                    "高相关标的对:"
                    + ",".join(f"{a}/{b} ρ={r:.2f}" for a, b, r in correlated)
                ),
            },
            evidence={"pairs": [
                {"a": a, "b": b, "spearman": round(r, 3)} for a, b, r in correlated
            ], "threshold": params.correlation_warn},
            suggestion={
                "en": (
                    "these names move together — diversification across them is "
                    "weaker than the symbol count suggests; consider replacing "
                    "one of each pair with a less-correlated candidate"
                ),
                "zh": (
                    "这些标的同涨同跌 — 实际分散度低于标的数量所暗示的水平;"
                    "可考虑将每对中的一只替换为相关性更低的候选"
                ),
            },
            rationale={
                "en": (
                    "Spearman rank correlation of daily returns over the replay "
                    "window (the live correlation library), robust to outliers"
                ),
                "zh": (
                    "回放窗口内日收益的 Spearman 秩相关(实盘同一相关性库),"
                    "对异常值稳健"
                ),
            },
        ))

    # ---- cash drag: signal scarcity, stated without overfitting advice.
    idle = sum(1 for c in cash_pct if c >= 99.0)
    if n > 0 and idle / n >= params.cash_drag_days_pct:
        items.append(AdviceItem(
            severity="INFO", code="CASH_DRAG",
            finding={
                "en": f"portfolio sat ≥99% in cash on {idle}/{n} bars ({idle / n:.0%})",
                "zh": f"组合在 {idle}/{n} 根K线上现金占比 ≥99%({idle / n:.0%})",
            },
            evidence={"idle_bars": idle, "total_bars": n},
            suggestion={
                "en": (
                    "consider widening the watchlist — NOT loosening entry "
                    "thresholds, which optimizes the past"
                ),
                "zh": (
                    "可考虑扩大自选列表 — 而不是放松入场阈值,"
                    "后者只是对过去做优化"
                ),
            },
            rationale={
                "en": (
                    "the §8 stack found few qualifying setups in these symbols "
                    "over this window; scarcity is information, not failure"
                ),
                "zh": (
                    "§8 决策栈在该时间窗内于这些标的上找到的合格形态很少;"
                    "稀缺本身是信息,不是失败"
                ),
            },
        ))

    return items
