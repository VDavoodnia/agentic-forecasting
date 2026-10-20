# td_one.uq — uncertainty quantification for TSFM forecasts

Conformal range layer for Chronos-2 (and TSFM) quantile forecasts.
Everything is PIT-safe by construction: a method sees only the *bank* of
past cases whose realized values were strictly observable before the
current origin (`origin' + h' − 1 < origin`).

| Module | What it is |
|---|---|
| `methods.py` | The UQ arms (exp74): native passthrough, fixed/rolling conformal, **CQR** (the default), weighted conformal, ACI, meta error models. |
| `features.py` | PIT-safe features for the meta error model: numeric (pre-origin), story (news pipeline), macro (Macro Signal Service, `signals_asof`). |
| `evaluate.py` | Rolling evaluation driver: builds cases, rebuilds the bank per origin, scores every arm. |
| `fusion.py` | exp76 fusion: F1 `MetaLevelCqr` (meta-informed conformal level; h≥3-gated form is the Food CPI champion), F2 conditional CQR (killed — cross-horizon score pooling bias). |
| `pooled.py` | exp81 cross-series pooled conformal banks: signed max-scores, side-specific width normalization, own-score shrinkage (n0=24), round-lag PIT pooling. Ships family-gated (≥8 series); per-series CQR elsewhere. |

Shipped configuration: per-series CQR default; pooled banks on
many-series families; F1 h3-gated fusion where story features and a
long bank exist; ACI as a regime-gated option. See
`experiments/exp74_report.md`, `exp76_report.md`, `exp81_report.md`.
