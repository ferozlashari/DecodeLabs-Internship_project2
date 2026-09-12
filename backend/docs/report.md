# Project 2 — Report: Data Classification Using AI

## 1. Task
Predict `OrderStatus` (Cancelled, Delivered, Pending, Returned, Shipped)
for an order, using the other columns in `Dataset_for_Data_Analytics.xlsx`
as input features.

## 2. Pipeline (IPO framework)

**INPUT**
- Loaded 1200 rows, 14 columns from the original Excel file.
- `CouponCode` had 309 missing values — these mean "no coupon was
  applied," so they were filled with the explicit label `"NoCoupon"`
  rather than dropped or guessed.
- Excluded `OrderID`, `CustomerID`, `TrackingNumber`,
  `ShippingAddress`, and `Date` as features — they are identifiers or
  free text; a KNN model can't generalize from unique IDs.

**PROCESS**
- Split 80/20 into train/test, stratified on `OrderStatus`.
- Numeric features (`Quantity`, `UnitPrice`, `ItemsInCart`,
  `TotalPrice`) scaled with `StandardScaler`.
- Categorical features (`Product`, `PaymentMethod`, `CouponCode`,
  `ReferralSource`) one-hot encoded.
- Model: `KNeighborsClassifier`, K configurable (default 5) — both from
  the CLI (`--k`) and live from the React frontend's slider.

**OUTPUT**
- Evaluated with accuracy, weighted F1, a full confusion matrix, and a
  per-class precision/recall/F1 report.

## 3. Results

| Metric | Value (K=5, 20% test) |
|---|---|
| Accuracy | ~0.17–0.18 |
| F1 (weighted) | ~0.17 |
| Random-guess baseline | ~0.20–0.25 (5 balanced classes) |

## 4. Why accuracy is low here (and why that's the honest answer)

This was checked against a `DummyClassifier` baseline and across a
sweep of K values (1 through 25) — performance stays flat and near or
below the random-guess baseline throughout. That pattern indicates
`OrderStatus` has no strong learnable relationship with the available
order details in this dataset (most likely because it's synthetic
data with the status assigned independently of the other columns,
rather than a real operational process behind it).

## 5. Takeaway
The project still demonstrates the full required skill set — loading
and understanding a dataset, a correct stratified train/test split,
applying a classification algorithm end-to-end (including a live,
interactive frontend), and validating the output honestly rather than
reporting a misleadingly polished number.
