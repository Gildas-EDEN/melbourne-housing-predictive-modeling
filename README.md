# Advanced Predictive Modeling & Market Segmentation

End-to-end machine learning project on the Melbourne housing market: data
engineering, unsupervised feature discovery, a model tournament for price
prediction, and a critical / ethical review of the results.

## Highlights

- **Data engineering**: missingness analysis via a nullity correlation matrix
  (Pearson correlation + p-values) to distinguish systematic from random
  missing values, then KNN imputation and outlier handling.
- **Unsupervised feature discovery**: dimensionality reduction with PCA, t-SNE
  and autoencoders (Keras); customer/property segmentation with K-Means and a
  business interpretation of each cluster.
- **Model tournament**: Linear Regression (baseline), Random Forest, Gradient
  Boosting, and a deep learning regressor (Keras), compared on RMSE / MAE / R².
- **Critical analysis**: performance audit and ethical review of the pipeline.

## Dataset

| | |
|---|---|
| **Name** | Melbourne Housing Market |
| **File** | `Melbourne_housing_FULL.csv` |
| **Source** | Kaggle — [Melbourne Housing Market](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market) (Tony Pino) |
| **License** | CC BY-NC-SA 4.0 (see Kaggle page) |
| **Rows / cols** | ~34,000 property sales, 21 columns |

### Reproducing the data step

The notebook loads the dataset directly from a raw URL, so no manual download is
required:

```python
import pandas as pd
url = "https://raw.githubusercontent.com/gildasedenakpo-sketch/Data_set_home/refs/heads/main/Melbourne_housing_FULL.csv"
df = pd.read_csv(url)
```

If you prefer a local copy, download `Melbourne_housing_FULL.csv` from Kaggle and
place it in `data/`.

## Project structure

```
.
├── notebooks/
│   └── advanced_predictive_modeling.ipynb   # main notebook (export from Colab)
├── data/                                     # local datasets (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Author

Gildas Edenakpo — M2 Econometrics & Data Science, Aix-Marseille School of Economics.
