# Scowtt MLE Take-Home Assessment

## Project Description
Build a single Python `.ipynb` notebook on Olist e-commerce data. Two predictions per user:
1. **Propensity Score** (0-1): probability of ordering in the next N days
2. **Predicted Conversion Value**: expected order dollar value

Deliverable: executed notebook with results, ready for interview discussion.

## Setup
- Python 3, venv: `python3 -m venv venv && source venv/bin/activate`
- Deps: `pip install pandas numpy matplotlib seaborn scikit-learn lightgbm jupyter notebook shap optuna`
- Dataset location: `datasets/`

---

## Workflow: Phase → Commit → Review → Next Phase

Each phase is a self-contained unit. After each phase: commit, user reviews, then proceed.

**Markdown cells:** Every phase includes markdown explanation cells at key decision points. These explain WHY, not just WHAT. They are interview-ready and should read as deliberate design choices.

---

## PHASE 1: Data Loading & EDA
**Commit after completion**

### Cell 1 (code): Imports
`pandas, numpy, matplotlib, seaborn, lightgbm, sklearn, shap`

### Cell 2 (markdown): Data Loading Strategy
> We load 8 of the 11 available tables. Three are excluded:
> - **marketing_qualified_leads** and **closed_deals** describe the seller acquisition funnel — they track how sellers joined the platform, not how customers purchase. Including them would conflate supply-side and demand-side behavior.
> - **geolocation** provides lat/lng for zip codes (~1M rows). Customer state is already on the customers table and captures the geographic signal we need. Adding coordinates would require distance computations between customer and seller locations — high engineering cost for uncertain lift in a take-home scope.
>
> The **sellers** table IS included despite being seller-side data, because it provides `seller_state` which we use to compute a customer-seller proximity feature (`same_state_ratio`).

### Cell 3 (code): Load all 8 CSVs with date parsing

### Cells 4–9 (code): Per-table EDA
For each table: `.shape`, `.dtypes`, `.describe()`, `.head(3)`, null counts + percentages, `.nunique()` on ID cols, `.value_counts()` on categoricals, key charts.

EDA highlights:
- customers: state distribution bar chart, customer_unique_id uniqueness
- orders: order_status value counts, NULL delivery dates for non-delivered
- order_items: price + freight histograms, items-per-order distribution
- payments: payment_type breakdown, installment distribution
- reviews: score distribution (bimodal: 58% 5★, 11.5% 1★), % with comments
- products: top 15 categories bar chart, missing dimensions/weight
- sellers: seller state distribution

### Cell 10 (markdown): Key Identifier Decision
> `customer_unique_id` has ~96K unique values vs ~99K `customer_id` values. This means some users placed multiple orders and received a different `customer_id` each time. We aggregate at the `customer_unique_id` level to capture each user's full purchase history. Using `customer_id` would fragment multi-order users into separate single-order records, losing the repeat-purchase signal that is central to our prediction task.

### Cell 11 (code): customer_id vs customer_unique_id analysis

### Cell 12 (code): Temporal analysis (date range, monthly volume plot, status distribution)

### Cell 13 (markdown): Missing Data Summary + Handling Strategy
> | Table | Column | % Missing | Strategy | Rationale |
> |---|---|---|---|---|
> | orders | order_approved_at | ~0.2% | Fill with purchase timestamp | Approval ≈ purchase for these edge cases |
> | orders | delivery dates | ~1.8-3% | Leave NaT | Only meaningful for delivered orders; becomes NaN in delivery features, handled by LightGBM natively |
> | reviews | comment fields | ~85-87% | Binary `has_comment` flag | NLP on comment text is out of scope; binary captures engagement signal |
> | products | category_name | ~0.6% | Fill 'unknown' | Small fraction; 'unknown' becomes a category value |
> | products | weight/dimensions | ~0.1% | Fill with median | Negligible missingness; median is stable |

### Cell 14 (markdown): Model Scope & Limitations
> **What this model CAN do:** Score existing customers (≥1 historical order) on their likelihood to repurchase and expected order value. This enables targeted re-engagement advertising.
>
> **What this model CANNOT do:** Identify potential first-time buyers. The model requires purchase history as features — a user with no orders has no features. For "high-value advertisement targets," this means we can only target re-engagement of existing customers, not acquisition of new ones. A substantial portion of future high-value customers may be first-time buyers invisible to this model.
>
> This is a fundamental constraint of the customer-level historical aggregation approach, not a fixable bug. Addressing it would require a different modeling paradigm (e.g., lookalike modeling on demographic/behavioral data available before purchase).

**8 tables to load:**
| File | Rows | Key Columns |
|---|---|---|
| `olist_customers_dataset.csv` | 99,441 | customer_id, customer_unique_id, customer_state, customer_city |
| `olist_orders_dataset.csv` | 99,441 | order_id, customer_id, 5 timestamp cols (parse as dates) |
| `olist_order_items_dataset.csv` | 112,650 | order_id, product_id, seller_id, price, freight_value |
| `olist_order_payments_dataset.csv` | 103,886 | order_id, payment_type, payment_installments, payment_value |
| `olist_order_reviews_dataset.csv` | 104,720 | order_id, review_score, review dates (parse as dates) |
| `olist_products_dataset.csv` | 32,951 | product_id, product_category_name, dimensions, weight, photos_qty, description_length |
| `olist_sellers_dataset.csv` | 3,095 | seller_id, seller_state, seller_city |
| `product_category_name_translation.csv` | 70 | Portuguese → English category mapping |

**Commit message pattern:** `Phase 1: Data loading and exploratory data analysis`

---

## PHASE 2: Data Aggregation & Order Master
**Commit after completion**

### Cell 15 (markdown): Aggregation Strategy
> Child tables (items, payments, reviews) have multiple rows per order. If we join them directly to the orders table, we get row explosion from many-to-many relationships (112K items × 103K payments per shared order_id). Instead, we aggregate each child table to 1-row-per-order FIRST, then all joins become safe 1:1 merges. This is the standard star-schema collapse pattern.

### Cell 16 (code): Enrich order_items
Join products + category_translation + sellers on respective keys.

### Cell 17 (markdown): What We Extract and Why
> **From products:** Category (translated to English), weight, volume (L×H×W), photo count, description length — these characterize what the user buys.
>
> **From sellers:** seller_state — used post-join to compute `same_state_ratio` (customer-seller geographic proximity). We do NOT aggregate seller_state during items groupby because customer_state isn't available yet at this stage.
>
> **Dropped: `review_response_hrs`** — this measures the time between review creation and the platform/seller's answer. It's a seller/platform responsiveness metric, not a user behavior. Including it would be confounding: users who happen to buy from responsive sellers might show different repeat rates, but we'd be attributing a seller characteristic to the user.

### Cell 18 (code): Aggregate child tables to order level

order_items_agg (group by order_id):
- `n_items`, `total_price`, `total_freight`, `n_categories`, `n_sellers`, `avg_weight`, `avg_volume`, `avg_photos_qty`, `avg_description_len`, `dominant_category`
- seller_state is NOT aggregated here — stays on enriched items for same_state computation later.

order_payments_agg (group by order_id):
- `total_payment`, `n_payment_methods`, `max_installments`, `primary_payment_type`, `used_voucher`

order_reviews_agg (group by order_id):
- `review_score`, `has_comment`

### Cell 19 (code): Build order_master via LEFT JOINs
```
orders ← customers (on customer_id) → gets customer_unique_id, customer_state
      ← order_items_agg (on order_id)
      ← order_payments_agg (on order_id)
      ← order_reviews_agg (on order_id)
```
Result: `order_master` — 99,441 rows, 1 per order, no duplication.

### Cell 20 (code): Compute same_state_ratio
After order_master is built:
1. Merge order_master (has `customer_state`) back to enriched items table (has `seller_state` per item)
2. Compute `customer_state == seller_state` per item row (boolean)
3. Aggregate to order level as `same_state_ratio = mean(boolean)`
4. Join back onto order_master

An order with 2 items from SP sellers + 1 from RJ, for an SP customer, correctly gets 0.67.

**Commit message pattern:** `Phase 2: Data aggregation and order master table`

---

## PHASE 3: Cutoff Validation, Temporal Split & User-Level Features
**Commit after completion**

### Cell 21 (markdown): Why We Need Cutoff Validation
> The entire model design assumes enough users repeat-purchase within the target window to train and evaluate on. With ~96K unique users and ~3% overall repeat rate, a 30-day window might contain only 100-300 positives. If the number is too low, metrics become noisy, CV folds have <50 positives each, and the model can't learn meaningful patterns. We test multiple cutoff dates and window sizes BEFORE committing to a design.

### Cell 22 (code): Validate target window viability (CRITICAL GATE)
For candidate cutoffs (2018-07-01, 2018-07-15, 2018-08-01) × window sizes (30/45/60 days):
- Count users with pre-cutoff orders who also order post-cutoff
- Print table: `cutoff × window_size → positive_count`
- Decision: ≥200 → proceed; <200 at 30-day → widen to 60; <100 at 60 → reframe problem

### Cell 23 (markdown): Order Status Filtering
> **`had_canceled_order`** is computed on unfiltered `order_master` — it explicitly looks for canceled/unavailable orders, so we need them present. After computing this flag per user, we filter:
>
> - **Feature aggregation** uses `order_status == 'delivered'` only. Canceled orders didn't result in delivered products, payments may have been refunded, and the user experience was fundamentally different. Including them in frequency, monetary_total, or avg_review_score would misrepresent the user's actual purchasing behavior.
> - **Label definition** uses `order_status == 'delivered'` only. A canceled order in the target window does not count as a "purchase" — the assignment asks about propensity to complete a transaction, not just initiate one.
>
> Document the count of excluded orders so the reader knows what was dropped (~2,963 non-delivered out of ~99,441 total).

### Cell 24 (markdown): Temporal Split Rationale
> We split by time, not randomly, to prevent data leakage. Features are built ONLY from orders before the cutoff date. Targets are defined ONLY by orders after the cutoff. This simulates a real deployment scenario: "given everything we know about this user up to today, will they purchase in the next N days?" Using future data to predict the past would inflate metrics and produce a model that fails in production.

### Cell 25 (code): Temporal split → feature_orders + label_orders
- Compute `had_canceled_order` per user on unfiltered order_master FIRST
- Filter to `order_status == 'delivered'` for both feature_orders and label_orders
- Print count of excluded non-delivered orders
- Leakage check: `assert feature_orders.order_purchase_timestamp.max() < cutoff`

### Cell 26 (markdown): Feature Engineering Decisions
> **36 features across 8 categories.** Key design choices:
>
> **`monetary_total` is kept** despite being correlated with `avg_order_value × frequency`. LightGBM (tree-based) is not affected by multicollinearity — trees split on one feature at a time and are immune to the instability that collinearity causes in linear models. We use SHAP values (not split-based importance) for interpretability, which correctly attributes importance across correlated features.
>
> **No sentinel values.** Features undefined for certain users (e.g., `avg_days_between_orders` for single-order users) are left as `NaN`. LightGBM handles missing values natively by learning the optimal split direction at each node. A sentinel value like -1 would create a misleading numeric threshold that the model splits on — effectively an indirect encoding of "has one order" that's redundant with `frequency=1` and `tenure_days=0`.
>
> **No label encoding for categoricals.** Label encoding (SP=1, RJ=2, MG=3...) implies ordinal relationships between categories that don't exist. Brazilian states aren't ordered. We use LightGBM's native categorical feature handling, which considers all possible partitions of category values at each split.
>
> **Dropped features:**
> - `review_response_hrs` — platform metric, not user behavior
> - `is_capital_state` — requires fuzzy matching of messy city names to 27 state capitals; high engineering cost for marginal lift
> - `score_x_delivery` — multiplication of unrelated scales (review 1-5 × delivery_delta -150 to +190); dominated by delivery_delta magnitude; LightGBM discovers interactions naturally through sequential splits

### Cell 27 (code): Aggregate to per-user level (36 features)

**36 features across 8 categories:**
| Category (count) | Features |
|---|---|
| RFM (7) | `recency_days`, `frequency`, `monetary_total`, `avg_order_value`, `max_order_value`, `total_freight`, `avg_items_per_order` |
| Product (6) | `n_categories`, `dominant_category`, `avg_product_weight`, `avg_product_volume`, `avg_photos_qty`, `avg_description_len` |
| Payment (5) | `primary_payment`, `avg_installments`, `max_installments`, `used_voucher`, `n_payment_methods` |
| Review (3) | `avg_review_score`, `min_review_score`, `left_comment` |
| Delivery (5) | `avg_delivery_days`, `avg_delivery_delta`, `ever_late`, `avg_approval_hrs`, `had_canceled_order` |
| Temporal (5) | `tenure_days`, `days_since_first`, `preferred_hour`, `preferred_dow`, `is_weekend_buyer` |
| Geographic (3) | `customer_state`, `n_sellers_used`, `same_state_ratio` |
| Interaction (2) | `freight_ratio`, `value_per_category` |

**Guards:**
- `freight_ratio`: `np.where(monetary_total > 0, total_freight / monetary_total, 0)` — prevents division by zero on voucher-only purchases
- `avg_approval_hrs`: cap at 99th percentile — max is ~4,509 hours (188 days), clearly erroneous; uncapped values distort SHAP plots

### Cell 28 (code): Construct target variables + print class counts
- `target_purchased` (binary): 1 if user orders in target period
- `target_order_value` (continuous): total payment in target period
- Print: "X positives out of Y total users (Z%)"

**Commit message pattern:** `Phase 3: Cutoff validation, temporal split, user-level features`

---

## PHASE 4: Model Training
**Commit after completion**

### Cell 29 (markdown): Train/Test Split Design
> The 80/20 split is a **random user-level split**, not a second temporal split. Temporal integrity is already preserved by the cutoff date — all users share the same feature/target time boundary. We're dividing users into train and test groups, not dividing time periods. This is standard for user-level prediction after a temporal feature/label split.
>
> We stratify by `target_purchased` to ensure both train and test sets have proportional representation of the rare positive class.

### Cell 30 (code): Train/test split (80/20 stratified) + feature prep
Set categorical columns to `category` dtype. NaNs are left as-is (LightGBM native handling).

### Cell 31 (markdown): Baseline Strategy
> We train two baselines before the full model, for different reasons:
>
> **Baseline 1 — Recency ranking:** A sanity check. Rank users by days since last order (most recent = highest score). If our ML model can't beat "recent buyers are more likely to buy again," something is fundamentally wrong. This is a strawman we expect to beat easily.
>
> **Baseline 2 — RFM Logistic Regression:** The real test. Three features (recency, frequency, monetary_total), logistic regression with balanced class weights. If LightGBM with 36 features can't convincingly beat 3-feature logistic regression, the extra complexity isn't justified — and that's an important finding worth discussing honestly, not hiding.
>
> The logistic regression uses only numeric features, so no categorical encoding is needed (no conflict with our "no label encoding" rule for LightGBM).

### Cell 32 (code): Baseline 1 — Recency ranking
### Cell 33 (code): Baseline 2 — RFM logistic regression

### Cell 34 (markdown): LightGBM Design Choices
> **`is_unbalance=True`**: Tells LightGBM to automatically adjust for class imbalance by weighting the minority class. With a positive rate of ~0.1-0.3%, this is essential — without it, the model would learn to predict 0 for everyone and achieve >99% accuracy.
>
> **Categorical features via native handling**: We pass `categorical_feature` to `.fit()` so LightGBM partitions categories optimally at each split, rather than relying on arbitrary numeric encodings.
>
> **5-fold stratified CV**: Reports AUC-ROC and AUC-PR as mean ± std across folds. **Caveat**: stratified random folds on data derived from temporally-ordered events may be slightly optimistic vs true forward-looking performance. In production, expanding-window time-series CV would be more rigorous. For a take-home, stratified CV is standard and acceptable.

### Cell 35 (code): LightGBM classifier + 5-fold CV
```python
lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
    num_leaves=31, is_unbalance=True, random_state=42)
```

### Cell 36 (markdown): Two-Stage Value Prediction
> We use a two-stage hurdle model:
> - **Stage 1**: Propensity model predicts P(purchase) for all users
> - **Stage 2**: Value model predicts order value *given that a purchase happens*
> - **Final expected value** = propensity × predicted_value_given_purchase
>
> This avoids the zero-inflated regression problem: training a regressor on all users (95%+ with target_value=0) creates a degenerate distribution where the model learns to predict ~0 for everyone.
>
> **Stage 2 approach depends on sample size:**
> - If ≥200 purchasers in target window → train LightGBM Regressor on purchasers only, compare RMSE to historical average, use whichever performs better
> - If <200 purchasers → use each user's historical `avg_order_value` as the prediction. Training a gradient boosting model on ~100 samples with 36 features will almost certainly overfit. A simple historical average is more robust and more defensible in an interview.

### Cell 37 (code): Conversion value model

**Commit message pattern:** `Phase 4: Baseline models and LightGBM training`

---

## PHASE 5: Evaluation & Diagnostics
**Commit after completion**

### Cell 38 (markdown): Evaluation Framework
> **Primary metric: AUC-PR (Average Precision)**, not accuracy or AUC-ROC.
>
> With a positive rate of ~0.1-0.3%, accuracy is meaningless — predicting 0 for everyone scores >99%. AUC-ROC can also be misleading at extreme imbalance because it weighs true negative rate heavily. AUC-PR focuses on precision and recall among predicted positives, which is what matters for ad targeting: of the users we'd target, what fraction actually converts?

### Cell 39 (code): Classification comparison table + ROC/PR curves (all 3 models)

### Cell 40 (markdown): Why Calibration Matters
> Propensity scores should be interpretable as probabilities. If the model says 0.05, roughly 5% of those users should actually purchase. Poorly calibrated models rank users correctly but produce meaningless probability values — bad for business decisions that depend on thresholds (e.g., "target everyone with >3% propensity").
>
> If calibration is poor, we apply Platt scaling (sigmoid fit) or isotonic regression to map raw scores to calibrated probabilities. We re-evaluate the calibration plot after correction.

### Cell 41 (code): Calibration plot + Platt scaling/isotonic if needed

### Cell 42 (code): Regression metrics (if regressor trained) + honest sample size caveat
> If regressor was trained on a small sample (<300 purchasers), note: "With N purchasers in the test set, RMSE/MAE/R² have wide confidence intervals and should be interpreted cautiously."

### Cell 43 (code): Score distributions (propensity histogram by label, expected value dist)

### Cell 44 (markdown): Feature Importance Approach
> We use SHAP values instead of LightGBM's default split-based importance. Split importance counts how often a feature is used in tree splits — but correlated features (like `monetary_total` and `avg_order_value`) split the importance between them, making both appear less important than they are. SHAP values correctly attribute marginal contribution and handle correlated features by measuring each feature's impact on individual predictions.
>
> After analyzing importance, we prune features with near-zero SHAP values and retrain. Showing that AUC barely changes demonstrates ML judgment: we evaluated feature contribution and simplified the model without sacrificing performance.

### Cell 45 (code): SHAP feature importance + pruning

**Commit message pattern:** `Phase 5: Model evaluation and diagnostics`

---

## PHASE 6: Final Output & Conclusions
**Commit after completion**

### Cell 46 (code): Per-user score table + top 20 ad targets
```python
final_output = DataFrame({
    'customer_unique_id': ...,
    'propensity_score': ...,          # 0-1 (calibrated)
    'predicted_conversion_value': ..., # $ (from regressor or historical avg)
    'expected_value': ...              # propensity × value
})
```

### Cell 47 (markdown): Conclusions
> - Key EDA findings
> - Model performance: LightGBM vs baselines (quantify the lift)
> - Honest discussion of limitations:
>   - Model only scores existing customers — cannot identify first-time buyer potential
>   - Positive class size and its impact on metric reliability
>   - Stage 2 regressor limitations (small sample or historical avg fallback)
>   - CV temporal mixing caveat
> - What we'd do differently with more time/data:
>   - Expanding-window time-series CV
>   - Hyperparameter tuning (Optuna/Bayesian)
>   - NLP on review text for sentiment features
>   - Lookalike modeling for first-time buyer targeting

---

## Critical Rules (apply to ALL phases)

### NaN Handling
**NO sentinel values. Ever.** Undefined features stay as `NaN`. LightGBM handles missing natively. A -1 sentinel creates a misleading split redundant with `frequency=1`.

### Categorical Encoding
**NO label encoding.** Use LightGBM native categorical handling:
- `customer_state` → `category` dtype
- `dominant_category` → `category` dtype
- `primary_payment` → `category` dtype

### Multicollinearity
**Keep `monetary_total`.** Trees don't care about collinearity. Use SHAP for interpretability.

### Features to AVOID (leakage/noise)
- `customer_id`, `order_id`, `product_id`, `seller_id` — high-cardinality IDs
- `zip_code_prefix` — 14K+ values, too sparse
- `product_name_length` — describes listing, not user
- Any post-cutoff feature — data leakage
- Raw timestamps — derive recency/tenure/hour instead
- `review_comment_message` text — binary `left_comment` captures 80% of signal
- `review_response_hrs` — platform metric, not user behavior

## Verification Checklist (end of project)
1. Cutoff validation passes: ≥200 positives (or plan adapted)
2. All cells run top-to-bottom in fresh kernel with no errors
3. No NaN sentinels anywhere
4. No label-encoded categoricals
5. Propensity scores in [0, 1], calibrated
6. No data leakage (features only from before cutoff)
7. LightGBM beats RFM LogReg (or honest discussion of why not)
8. Feature pruning demonstrates simplification
9. Final table: one row per `customer_unique_id`
