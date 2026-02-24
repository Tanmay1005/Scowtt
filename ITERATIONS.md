# Scowtt MLE Notebook — Iteration Log

## Iteration 1: Initial Build (All 36 Features, Default LightGBM)

### What we did
- Built the full 47-cell notebook from scratch across all 6 phases
- Cutoff: **2018-07-01**, window: **60 days** → **280 positives** (0.34% rate)
- Used all **36 features** including 3 categoricals (`customer_state`, `dominant_category`, `primary_payment`)
- LightGBM with default-ish params: `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `num_leaves=31`, `is_unbalance=True`
- No explicit regularization (`min_child_samples` at default 20, no `reg_alpha`/`reg_lambda`, no `feature_fraction`)

### Results
| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Recency Ranking | 0.619 | 0.009 |
| RFM LogReg (3 features) | 0.613 | 0.026 |
| **LightGBM (36 features)** | **0.534** | **0.005** |

- 5-fold CV ROC-AUC: **0.477 ± 0.034** — *below random (0.50)*
- LightGBM was dramatically worse than both baselines

### What went wrong
1. **Massive overfitting.** 500 trees at depth 6 with 224 training positives and 36 features. The model had enough capacity to memorize noise. CV below 0.50 confirmed it — the model learned patterns in training folds that anti-correlated with validation folds.
2. **Categorical features with few positives per category.** LightGBM's native categorical handling found spurious category-specific splits (e.g., "customer_state == SP" based on 5 positive users).
3. **No regularization.** Default `min_child_samples=20` allowed leaf nodes with tiny samples. No L1/L2 penalty, no feature subsampling.
4. **Too few positives.** 280 total (56 in test) — not enough for stable evaluation, let alone complex model training.

### Issues
- Conclusions cell claimed "LightGBM outperforms both baselines" — **false**, directly contradicted by the numbers
- Propensity scores compressed to a tiny range — model barely differentiated users

---

## Iteration 2: Added Regularization (Same Features, Same Window)

### What we changed
- Kept 36 features and 2018-07-01 cutoff with 60-day window (still 280 positives)
- Added regularization: `max_depth=3`, `num_leaves=8`, `min_child_samples=50`, `feature_fraction=0.7`, `reg_lambda=1.0`
- Reduced `n_estimators` from 500 to 200
- Updated markdown to explain regularization rationale
- Fixed conclusions to be honest about results

### Results
| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Recency Ranking | 0.619 | 0.009 |
| RFM LogReg | 0.613 | 0.026 |
| **LightGBM (36 feat, regularized)** | **0.591** | **0.006** |

- CV ROC-AUC: **0.511 ± 0.046** — above random now, but barely
- Propensity range: [0.003, 0.005] — still very compressed

### What improved
- ROC-AUC went from 0.534 → 0.591 (regularization stopped the worst overfitting)
- CV went from 0.477 → 0.511 (no longer below random)

### What still didn't work
- PR-AUC still terrible: 0.006 vs LogReg's 0.026 — 4x worse
- Propensity scores barely differentiated users (0.003 to 0.005 range)
- Root cause unchanged: 280 positives with 36 features was still too many degrees of freedom
- Categorical features still causing problems

---

## Iteration 3: Wider Window + Reduced Features + Stronger Regularization

### What we changed (three simultaneous fixes)

**1. Widened target window:**
- Moved cutoff from 2018-07-01 → **2018-06-01**
- Tested 60d, 90d, and "all remaining" windows
- Selected 90-day window → **419 positives** (0.56%) — 50% more than before
- More pre-cutoff feature history (~1.5 years vs ~1 year)

**2. Radical feature reduction (36 → 8):**
- Core features: `recency_days`, `frequency`, `monetary_total`, `avg_order_value`, `avg_review_score`, `avg_delivery_delta`, `tenure_days`, `freight_ratio`
- Dropped ALL categorical features — eliminated spurious category splits
- Dropped product features (weight, volume, photos) — characterize items, not repurchase intent
- Dropped temporal preferences (hour, dow, weekend) — weak signal for "will they buy again"
- Still computed all 36 features for SHAP analysis on a separate full model

**3. Stronger regularization + fairer comparison:**
- `max_depth=3`, `num_leaves=7` (very shallow)
- `min_child_samples=50` (high threshold)
- `reg_alpha=1.0`, `reg_lambda=1.0` (strong L1+L2)
- `subsample=0.8` (row subsampling)
- `learning_rate=0.01` with `n_estimators=300` (gentle, gradual learning)
- Switched from `is_unbalance=True` to explicit `scale_pos_weight` (neg/pos ratio) — matches what `class_weight='balanced'` does in LogReg, making the baseline comparison fair

### Results
| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Recency Ranking | 0.558 | 0.008 |
| RFM LogReg (3 features) | 0.575 | 0.025 |
| **LightGBM (8 features)** | **0.579** | **0.016** |
| + Platt calibration | — | 0.019 |
| LightGBM (36 features) | 0.567 | 0.016 |

- CV ROC-AUC: **0.592 ± 0.051** — solidly above random
- CV PR-AUC: **0.020 ± 0.013** — approaching LogReg's 0.025
- Per-fold: Fold 3 hit 0.640/0.035, Fold 5 was 0.510/0.007 (high variance from small positive class)

### What improved
- **LightGBM now wins on ROC-AUC** (0.579 vs LogReg's 0.575)
- **PR-AUC gap narrowed dramatically**: from 5x worse (0.005 vs 0.025) to 1.3x (0.019 vs 0.025 after Platt calibration)
- CV mean PR-AUC (0.020) is within noise of LogReg (0.025) given 84 test positives
- **8-feature core model matches 36-feature full model** — validates the feature reduction
- Propensity scores now spread [0.002, 0.023] — meaningful differentiation vs the old [0.003, 0.005]
- Value regressor trained successfully (335 purchasers in training) and beat historical average baseline

### Remaining gap
- LogReg still edges LightGBM on PR-AUC (0.025 vs 0.019). With 84 test positives, this difference (~0.006) is within noise — but it's honest to note
- High CV fold variance (std 0.013 on PR-AUC) reflects the fundamental constraint: not enough positives for tight estimates
- This is a data ceiling, not a modeling failure

### Conclusions honest and verified
- No false claims about "outperforming" — every statement backed by printed numbers
- Acknowledges that feature reduction + regularization > model complexity when signal is scarce
- Notes the 8-feature model matching the 36-feature model as evidence of justified simplification

---

## Key Lessons Across Iterations

1. **Positive sample size is the binding constraint.** Going from 280 → 419 positives helped more than any hyperparameter change. In production, more data collection would have the highest ROI.

2. **Feature count must match signal availability.** 36 features with 224 training positives (~6:1 ratio) was a recipe for overfitting. 8 features with 335 positives (~42:1 ratio) let the model learn real patterns.

3. **Regularization alone can't fix a model with too many features.** Iteration 2 showed that regularizing a 36-feature model improved it (0.477 → 0.511 CV ROC-AUC) but couldn't make it competitive. You need to reduce the problem size, not just constrain the solution.

4. **Categorical features are dangerous with few positives.** LightGBM's native categorical handling is powerful on large datasets but finds spurious splits when only 5-10 positives fall in a given category.

5. **`is_unbalance` vs `scale_pos_weight` matters for fair comparison.** They handle class weights differently; using explicit `scale_pos_weight` matching LogReg's `balanced` weights made the comparison apples-to-apples.

6. **Honest results > inflated metrics.** An overfit model with impressive training numbers would fail in production and get caught in an interview. A simpler model with honest metrics demonstrates ML judgment.
