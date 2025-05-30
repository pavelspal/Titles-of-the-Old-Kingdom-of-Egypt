# Folder scr
This folder includes source code and data for modeling.

## Folder Structure

```plaintext
├── data/                      # Sample or anonymized data files
├── jupyter_notebook/         # Jupyter notebooks for analysis and modeling
│   ├── R_script/             # R Jupyter notebooks
│   └── *.ipynb               # Python Jupyter notebooks
├── supp/                     # General support functions used across notebooks and scripts
└── *.py                      # Python scripts
```

## Nested Folder Overview

| Folder Name | Description |
|---|---|
| data | Includes input data and output summaries |
| jupyter_notebook | Jupyter notebooks for analysis and modeling |
| supp | General support functions used across notebooks and scripts |

## File Overview

| File Name | Description |
|---|---|
| analyze_R_log.py | Converts logistic coefficients into a CSV file |
| pickle_analyzer.py | Summarizes tables from the Maatbase database |
| summary_all.py | Sequentially runs post-model summaries |
| summary_model.py | Summarizes all models |
| summary_persons.py | Post-model summary of persons |
| summary_plot_shap_mean.py | Generates SHAP mean plots for all models |
| summary_plot_shap_summary.py | Generates SHAP summary plots for all models |
| summary_shap_calculator.py | Calculates SHAP values for all models |
| summary_title.py | Post-model summary of titles |
| title_correlation.py | Inspects title correlations |
| unpack_df_all.py | Unpacks person features into one-hot encoding |
| unpack_merge_df.py | Merges features with the target feature |
| unpack_vizier.py | Creates target mapping |
| README.md | This folder overview (this file) |
