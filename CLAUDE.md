# Scowtt MLE Take-Home Assessment

## Project Description
Build a single Python `.ipynb` notebook on Olist e-commerce data. Two predictions per user:
1. **Propensity Score** (0-1): probability of ordering in the next N days
2. **Predicted Conversion Value**: expected order dollar value

Deliverable: executed notebook with results, ready for interview discussion.

## Setup
- Python 3, venv: `python3 -m venv venv && source venv/bin/activate`
- Deps: `pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost jupyter notebook shap`
- Dataset location: `datasets/`

---

## Final Notebook Structure (60 cells, after 9 iterations)

### Phase 1: Data Loading & EDA (Cells 0-13)
- **Cell 0**: Imports (pandas, numpy, matplotlib, seaborn, lightgbm, xgboost, catboost, sklearn, shap, scipy)
- **Cell 1**: Data loading strategy markdown (8 of 11 tables, why 3 excluded)
- **Cell 2**: Load all 8 CSVs with date parsing
- **Cells 3-8**: Per-table EDA (customers, orders, items, payments, reviews, products/sellers)
- **Cell 9**: Key identifier decision markdown (`customer_unique_id` vs `customer_id`)
- **Cell 10**: customer_id vs customer_unique_id analysis
- **Cell 11**: Temporal analysis
- **Cell 12**: Missing data summary markdown
- **Cell 13**: Model scope & limitations markdown

### Phase 2: Data Aggregation (Cells 14-19)
- **Cell 14**: Aggregation strategy markdown
- **Cell 15**: Enrich order_items (products + categories + sellers)
- **Cell 16**: What we extract and why markdown
- **Cell 17**: Aggregate child tables to order level
- **Cell 18**: Build order_master via LEFT JOINs
- **Cell 19**: Compute same_state_ratio

### Phase 3: Cutoff, Split & Features (Cells 20-29)
- **Cell 20**: Why we need cutoff validation markdown
- **Cell 21**: Cutoff validation (expanded range, multiple cutoffs x windows)
- **Cell 22**: Window sensitivity analysis (30/60/90/182-day at March 2018)
- **Cell 23**: Window sensitivity markdown (182-day = reactivation propensity compromise)
- **Cell 24**: Order status filtering markdown
- **Cell 25**: Temporal split rationale markdown
- **Cell 26**: Temporal split code (March 2018 cutoff, 182-day window, 654 positives)
- **Cell 27**: Feature engineering decisions markdown (15 core features)
- **Cell 28**: Per-user aggregation (RFM + momentum + interaction features)
- **Cell 29**: Construct targets + print class counts

### Phase 4: Model Training (Cells 30-44)
- **Cell 30**: Train/test split markdown
- **Cell 31**: Train/test split (80/20 stratified) + 3-fold OOF target encoding
- **Cell 32**: Baseline strategy markdown
- **Cell 33**: Baseline 1 — Recency ranking
- **Cell 34**: Baseline 2 — RFM Logistic Regression
- **Cell 35**: LightGBM design choices markdown (15 features, strong regularization)
- **Cell 36**: LightGBM base classifier + stratified CV + time-series CV
- **Cell 37**: Model shootout markdown (LightGBM vs XGBoost, CatBoost excluded)
- **Cell 38**: Model shootout code (XGBoost with early stopping)
- **Cell 39**: Grid + randomized search markdown (custom safe scorer)
- **Cell 40**: Grid search (4 combos) + randomized search (10 combos) + XGBoost full train
- **Cell 41**: Log-odds blend markdown
- **Cell 42**: Blend code + final model selection (base LightGBM wins)
- **Cell 43**: Rolling cutoff backtest markdown
- **Cell 44**: Rolling backtest code (4 cutoffs x 60-day windows)

### Phase 4b: Value Prediction (Cells 45-46)
- **Cell 45**: Hierarchical value prediction markdown
- **Cell 46**: Hierarchical value prediction code (user avg -> cohort avg -> global avg)

### Phase 5: Evaluation & Diagnostics (Cells 47-55)
- **Cell 47**: Evaluation framework markdown (PR-AUC primary metric)
- **Cell 48**: Comparison table + ROC/PR curves (7 models, with Brier)
- **Cell 49**: Calibration matters markdown
- **Cell 50**: Calibration plot + Platt scaling + Brier pre/post
- **Cell 51**: Value prediction metrics (historical avg)
- **Cell 52**: Lift/decile analysis + Precision@K + Recall@K
- **Cell 53**: Score distributions
- **Cell 54**: SHAP analysis markdown
- **Cell 55**: SHAP feature importance

### Phase 6: Final Output & Conclusions (Cells 56-59)
- **Cell 56**: Per-user score table + top 20 ad targets
- **Cell 57**: Productionization notes markdown
- **Cell 58**: Iteration log (Appendix — all 9 iterations summarized)
- **Cell 59**: Conclusions markdown (model comparison, limitations, next steps)

---

## Final Model Configuration

### Data
- **Cutoff**: 2018-03-01
- **Window**: 182 days (reactivation propensity within 6 months)
- **Positives**: 654 (1.18% rate)
- **Users**: 55,525 with pre-cutoff orders

### 15 Core Features
| Category | Features |
|---|---|
| RFM (3) | `recency_days`, `monetary_total`, `avg_order_value` |
| Interaction (3) | `purchase_velocity`, `freight_ratio`, `value_per_category` |
| Decay (2) | `monetary_decayed`, `frequency_decayed` |
| Encoded (2) | `customer_state_encoded`, `dominant_category_encoded` (3-fold OOF target encoding, smoothing=20) |
| Momentum (3) | `avg_days_between_orders`, `avg_review_delay_days`, `spending_trend` |
| Temporal (1) | `tenure_days` |
| Delivery (1) | `avg_delivery_delta` |

### Model
- **Algorithm**: LightGBM (`n_estimators=300, learning_rate=0.01, max_depth=3, num_leaves=7, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0, subsample=0.8, scale_pos_weight=neg/pos ratio`)
- **Target encoding**: 3-fold OOF (each training row encoded by OTHER folds, test rows by full training set)
- **Calibration**: Platt scaling (fitted on training data)
- **Value prediction**: Hierarchical historical average (user -> cohort -> global), no regressor

### Results
| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Recency Ranking | 0.561 | 0.014 |
| RFM LogReg (3 features) | 0.588 | 0.045 |
| **LightGBM Base (15 features)** | **0.604** | **0.047** |
| XGBoost | 0.605 | 0.045 |
| LightGBM Grid Search | 0.605 | 0.044 |
| LightGBM Random Search | 0.599 | 0.045 |
| Blend (raw, w=0.55) | 0.609 | 0.046 |

- **Top decile lift**: 2.44x
- **Top 10% captures**: 24.4% of converters
- **Brier**: 0.196 raw, 0.012 Platt-calibrated
- **Backtest**: PR-AUC 0.021 +/- 0.009 across 4 cutoffs (Jan-Apr 2018)

---

## Key Decisions Made Across 9 Iterations

1. **No Optuna/adaptive HPO** — overfits with ~100 positives per CV fold (tested 3 times, always hurt test PR-AUC)
2. **No CatBoost in final comparison** — its regularization scales differ from LightGBM, unfair without separate tuning
3. **No LightGBM early stopping** — PR-AUC too noisy per-fold (stopped at iteration 1)
4. **No value regressor** — negative R-squared across iterations, historical average is correct
5. **3-fold OOF > 5-fold OOF** — more data per fold = less noisy encoding with scarce positives
6. **Base LightGBM > search models** — grid and randomized search confirmed manual params near-optimal
7. **Selection threshold** — search model must beat base by >0.001 PR-AUC to be selected (prevents noise-driven selection)

---

## Critical Rules

### NaN Handling
**NO sentinel values. Ever.** Undefined features stay as `NaN`. LightGBM handles missing natively.

### Target Encoding
**3-fold OOF** for training rows (each row encoded using OTHER folds only). Test rows use full training set map. `smoothing=20`, `min_count=5`. No self-label leakage.

### Multicollinearity
**Keep `monetary_total`.** Trees don't care about collinearity. Use SHAP for interpretability.

### Features to AVOID (leakage/noise)
- High-cardinality IDs (`customer_id`, `order_id`, `product_id`, `seller_id`)
- `zip_code_prefix` — 14K+ values, too sparse
- Any post-cutoff feature — data leakage
- Raw timestamps — derive recency/tenure instead
- `review_comment_message` text — binary `left_comment` captures signal
- `review_response_hrs` — platform metric, not user behavior
- `late_delivery_ratio` — zero SHAP, pruned in iter 7
- `ordered_last_30d`, `ordered_last_90d` — zero SHAP, pruned in iter 6
- `primary_payment_encoded` — near-zero SHAP, pruned in iter 6

## Verification Checklist
1. All 60 cells run top-to-bottom in fresh kernel with no errors
2. No NaN sentinels anywhere
3. OOF target encoding (no self-label leakage)
4. Propensity scores in [0, 1], Platt-calibrated
5. No data leakage (features only from before cutoff)
6. LightGBM beats recency baseline; matches/beats LogReg on PR-AUC
7. Grid + random search validate base params are near-optimal
8. Final table: one row per `customer_unique_id`
9. Iteration log in notebook documents full journey
10. Conclusions honest about limitations and data-driven signal ceiling
