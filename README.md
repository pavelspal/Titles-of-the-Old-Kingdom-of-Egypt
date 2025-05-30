# Detection of Relationships Between Title of the Old Kingdom of Egypt

This repository provides a overview of the code and selected visualizations created as part of my diploma thesis:

**Diploma Thesis Title:** *Detection of Relationships Between Titles of the Old Kingdom of Egypt*  
**Author:** Pavel Stojaspal  
**Year:** 2025

## Abstract

This master thesis aims to analyze the patterns behind the vizier and other titles of the Old Kingdom of Egypt. As a data source we will use unique Maatbase database which was created under the Czech Institute of Egyptology. By inspecting persons together with their titulary, we create two datasets that we will use in our modelling. Later, by means of logistic regression and multilayer perceptron, we implement twelve models to analyze the patterns. The model analysis is performed from multiple perspectives. Firstly, we evaluate the performance of the models. Secondly, we highlight persons with high predictions although these persons were not viziers. Finally, we use SHAP values to analyze the importance of the features in each model. The results are compared with other publications. As a by-product we show a projection of titles categories onto the first two principal component, where family category is easily distinguished from other categories. This thesis is concluded with Attachment that present multiple tables and figures that complement our results in more details.

## Key Words

Old Kingdom of Egypt, vizier, machine learning, logistic regression, multilayer perceptron, SHAP values.

## Thesis PDF
The thesis can be viewed in the PDF [dp_pavel_stojaspal](https://github.com/pavelspal/Titles-of-the-Old-Kingdom-of-Egypt/blob/main/dp_pavel_stojaspal.pdf).

## Main Files

| File Name | Path | Description |
|---|---|---|
| [dp_pavel_stojaspal.pdf](https://github.com/pavelspal/Titles-of-the-Old-Kingdom-of-Egypt/blob/main/dp_pavel_stojaspal.pdf) |  | Thesis PDF |
| [log_regression_v1.ipynb](https://github.com/pavelspal/Titles-of-the-Old-Kingdom-of-Egypt/blob/main/scr/jupyter_notebook/R_script/log_regression_v1.ipynb) | scr/jupyter_notebook/R_script | Logistic models for feature version 1 |
| [log_regression_v2.ipynb](https://github.com/pavelspal/Titles-of-the-Old-Kingdom-of-Egypt/blob/main/scr/jupyter_notebook/R_script/log_regression_v2.ipynb) | scr/jupyter_notebook/R_script | Logistic models for feature version 2 |
| [nn_copy_log_regression_v1_weight_50.ipynb](https://github.com/pavelspal/Titles-of-the-Old-Kingdom-of-Egypt/blob/main/scr/jupyter_notebook/nn_copy_log_regression_v1_weight_50.ipynb) | scr/jupyter_notebook | MLP models for feature version 1 |
| [nn_copy_log_regression_v2_weight_50.ipynb](https://github.com/pavelspal/Titles-of-the-Old-Kingdom-of-Egypt/blob/main/scr/jupyter_notebook/nn_copy_log_regression_v2_weight_50.ipynb) | scr/jupyter_notebook | MLP models for feature version 2 |
|  |  |  |
|  |  |  |
|  |  |  |

## Repository Structure

```plaintext

├── src/                        # Source code and data for modeling
  ├── data/                     # Sample or anonymized data files
  ├── jupyter_notebooks/        # Jupyter notebooks for analysis and modeling
  ├── supp/                     # General support functions used across notebooks and scripts
  └── *.py/                     # Python scripts
├── dp_pavel_stojaspal.pdf      # PDF with the full thesis
├── dp_zadani.pdf               # PDF wit hte thesis assignment
├── environment.yml             # YML file containing a list of required Python and R packages
├── requirements.txt            # TXT file containing a list of required Python and R packages
└── README.md                   # Project overview (this file)
```

## Clone Repository and Download Requirements

This project uses an Anaconda environment to manage both Python and R dependencies. To use the same dependecies use `environment.yml` or  `requirements.txt`.

Clone repository:
```bash
git clone https://github.com/pavelspal/Titles-of-the-Old-Kingdom-of-Egypt.git
cd Titles-of-the-Old-Kingdom-of-Egypt
```

Create the environment from `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate titles_env
```
