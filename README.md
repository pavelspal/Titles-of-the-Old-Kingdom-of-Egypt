# Detection of Relationships Between Titles of the Old Kingdom of Egypt

This repository provides a brief overview of the code and selected visualizations created as part of my diploma thesis:

**Diploma Thesis Title:** *Detection of Relationships Between Titles of the Old Kingdom of Egypt*  
**Author:** Pavel Stojaspal  
**Year:** 2025

## Abstract

This master thesis aims to analyze the patterns behind the vizier and other titles of the Old Kingdom of Egypt. As a data source we will use unique Maatbase database which was created under the Czech Institute of Egyptology. By inspecting persons together with their titulary, we create two datasets that we will use in our modelling. Later, by means of logistic regression and multilayer perceptron, we implement twelve models to analyze the patterns. The model analysis is performed from multiple perspectives. Firstly, we evaluate the performance of the models. Secondly, we highlight persons with high predictions although these persons were not viziers. Finally, we use SHAP values to analyze the importance of the features in each model. The results are compared with other publications. As a by-product we show a projection of titles categories onto the first two principal component, where family category is easily distinguished from other categories. This thesis is concluded with Attachment that present multiple tables and figures that complement our results in more details.

## Key Words

Old Kingdom of Egypt, vizier, machine learning, logistic regression, multilayer perceptron, SHAP values.

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
