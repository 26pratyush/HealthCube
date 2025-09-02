# Results & Limitations

## Evaluation Results

Evaluation was conducted on **624 patient records**.

### K-value (Regression)

* **MAE** = 0.380 mmol/L
* **RMSE** = 0.473 mmol/L
* **Within ±0.20 mmol/L**: 29.5%
* **Within ±0.10 mmol/L**: 16.7%

These results show that the model **struggles to precisely estimate serum potassium levels**. Only about one-third of predictions fall within the clinically tight ±0.20 mmol/L range. This limits its direct use for exact K estimation.

---

### Severity Classification

* **Accuracy** = 94.1%
* **Macro F1** = 0.892

**Class-wise performance:**

* **Normal**: 429/441 correct (95%+)
* **Moderate Hypo**: 123/140 correct (88%)
* **Severe Hypo**: 19/24 correct (79%)
* **Severe Hyper**: 10/11 correct (91%)
* **Mild Hyper**: 6/8 correct (75%)

**Severity classes are predicted much more reliably** than raw K values. The model clearly distinguishes between normal, moderate, and severe states, which are clinically more relevant for decision-making than exact mmol/L predictions.

---

## Limitations

While the severity classification is robust, the regression accuracy remains limited due to **dataset size**, **signal variability**, and **borderline cases** (e.g., K=3.4 vs K=3.5). Future improvements should focus on richer features and specialized regression models, while severity classification can already be considered a reliable screening tool.
