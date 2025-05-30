# Folder scr/jupyter_notebook
This folder includes Jupyter notebooks for analysis and modeling.

## Folder Structure

```plaintext
├── R_script/                    # R scripts
│   ├── R_functions/             # R support functions
│   ├── R_log/                   # R summary outputs
│   └── *.ipynb                  # R Jupyter notebooks
└── *.ipynb                      # Python Jupyter notebooks
```

## File Overview

| File Name | Description |
|---|---|
| analyze_roc.ipynb | Inspection of ROC curves for all models |
| analyze_title_plot.ipynb | Plots post-model title summary |
| data_inspection_title_table.ipynb | Inspection of title mapping |
| database_schema_inspection.ipynb | Inspection of database primary keys |
| database_tables_overview.ipynb | Database table overview |
| df_all_person_all.ipynb | Merge information about all persons |
| df_all_person_tomb_all.ipynb | Merge information about persons and tombs |
| df_all_tomb_all_.ipynb | Merge information about all tombs |
| df_encoding_dynasty.ipynb | Encodes dynasty into numerical and categorical features |
| df_encoding_father_was_vizier.ipynb | Encodes whether the father of a given person was vizier |
| df_encoding_title_to_clusters.ipynb | Encodes titles into clusters |
| df_encoding_tittle_cluster_to_PCA.ipynb | Performs PCA on title clusters |
| df_encoding_tittle_to_PCA.ipynb | Performs PCA on titles |
| df_feature_split_dynasty_categorical.ipynb | Splits data into train, validation, and test sets for categorical dynasty |
| df_feature_split_dynasty_numerical.ipynb | Splits data into train, validation, and test sets for numerical dynasty |
| df_feature_v1.ipynb | Defines feature version 1 |
| df_feature_v2_conditional_prob.ipynb | Defines titles for feature version 2 |
| df_feature_v2_dynasty_categorical.ipynb | Defines feature version 2 with categorical dynasty |
| df_feature_v2_dynasty_numerical.ipynb | Defines feature version 2 with numerical dynasty |
| df_map_r_title.ipynb | Creates map between R and Python title names |
| glm_model_analyzer.ipynb | Post-model analysis |
| nn_copy_log_regression_v1.ipynb | Creates MLP models for feature version 1 |
| nn_copy_log_regression_v1_weight_50.ipynb | Creates MLP models for feature version 1 with weighted loss |
| nn_copy_log_regression_v2.ipynb | Creates MLP models for feature version 2 |
| nn_copy_log_regression_v2_weight_50.ipynb | Creates MLP models for feature version 2 with weighted loss |
| pca_2d_title_cluster_eigenvalues.ipynb | PCA on title clusters (eigenvalues) |
| pca_2d_title_cluster_hack.ipynb | PCA on title clusters (scaling) |
| pca_2d_title_cluster_v2.ipynb | PCA on title clusters |
| pca_title_cluster_normalization.ipynb | PCA on title clusters (normalization) |
| set_path.py | Support file |
| supp_merge.py | Support file |
| README.md | This folder overview (this file) |
