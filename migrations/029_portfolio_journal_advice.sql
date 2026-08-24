-- 029_portfolio_journal_advice.sql — explainability for portfolio backtests
-- (user mandate 2026-08-20: "自动持仓调仓要有可解释性…结合风控算法模型,
-- 对组合进行建议,并说明理由").
--
-- ADDITIVE: two JSONB columns on 028's portfolio_backtests, mirrored in
-- apps/gateway/db.py::PortfolioBacktestRecord in migration order (mirror
-- rule; tests/test_migration_parity.py registers this file as the ALTER
-- extension of 028).
--
-- `journal` — the rebalance journal: every ENTER (with the full sizing
-- arithmetic: tier budget × prior equity ÷ stop, then each cap that
-- trimmed the quantity), every EXIT (the shared exit engine's rule
-- verbatim), and every SKIP (the capital constraint that crowded a
-- selected candidate out). This is the record that distinguishes
-- "the matrix said no" from "capital said no".
--
-- `advice` — deterministic risk-model findings over the replay (the SAME
-- live libraries: historical VaR/ES with method labels, §2.8 drawdown,
-- Spearman correlation), each with severity/evidence/suggestion/rationale.

ALTER TABLE portfolio_backtests ADD COLUMN IF NOT EXISTS journal JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE portfolio_backtests ADD COLUMN IF NOT EXISTS advice  JSONB NOT NULL DEFAULT '[]'::jsonb;
