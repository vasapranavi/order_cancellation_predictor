#  Order Cancellation Predictor  

A machine learning project to predict whether an online order will be **canceled** or **not canceled**, with a focus on handling **class imbalance** and maximizing **recall** for the minority class.  

---

##  Overview  
- **Goal**: Anticipate cancellations to reduce inefficiencies and improve customer experience.  
- **Target variable**:  
  - `Canceled` → 1  
  - `Not_Canceled` → 0  
- **Challenge**: Only ~3.7% of orders are canceled (highly imbalanced).  

---

## Approach  
1. **Preprocessing**  
   - Handle missing values, outliers, and scaling.  
   - One-Hot Encode low-cardinality categoricals; frequency encode high-cardinality ones.  
   - Feature engineering from `slot_date`, price, basket size, etc.  

2. **Class Imbalance**  
   - Used **SMOTE** to upsample minority class in training.  

3. **Models Evaluated**  
   - Logistic Regression → highest recall.  
   - LightGBM → highest F1 score.  
   - Random Forest & XGBoost also tested.  

4. **Ensemble**  
   - Weighted soft voting of **Logistic Regression + LightGBM**.  
   - Higher weight to Logistic Regression for recall.  
   - Custom threshold (0.4) tuned for better minority detection.  

---

## Results  

| Model               | Recall | F1 Score |
|----------------------|--------|----------|
| Logistic Regression  | 0.62   | 0.57     |
| Random Forest        | 0.32   | 0.64     |
| XGBoost              | 0.45   | 0.66     |
| LightGBM             | 0.49   | 0.69     |
| **Ensemble (LR + LGBM)** | ↑ Recall | Competitive F1 |

---
## Production Architecture
Following is the proposed production level architecture

<img width="385" height="512" alt="20250915_0006_Recommendations Workflow Diagram_remix_01k557hf3kepraev8k7egdebzd" src="https://github.com/user-attachments/assets/74929f49-e51e-4083-a6ad-c06a3fbddbbd" />

---

## Future Work  
- Try stacking/blending ensembles.  
- Add external features (holidays, promotions).  
- Use probability calibration for better decision thresholds.  
- Deploy with monitoring for data drift.  

---

## Requirements  

```bash
pip install lightgbm xgboost imbalanced-learn scikit-learn pandas matplotlib seaborn
```

---

## Usage  

```bash
jupyter notebook order_cancellation_predictor_.ipynb
```

- Run preprocessing and training.  
- Evaluate validation results.  
- Generate test predictions (`status` column).  
- Save results:  

```python
test.to_csv("order_cancellation_predictions.csv", index=False)
```

---
