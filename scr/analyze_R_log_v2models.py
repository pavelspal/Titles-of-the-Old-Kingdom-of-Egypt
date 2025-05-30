import pandas as pd
import re
import os

import pandas as pd

from supp.support_load import get_path
from supp.support_constants import PATH_R_LOG, PATH_MODEL_OVERVIEW
from supp.support_save import save_df
from supp.support_parse_R_log import parse_file

"""
--------------------------------
    FILE BIO
--------------------------------
- file_name: analyze_R_log.py
- motivation:
    jupyter notebooks under 'jupyter_notebook\R_script' contain logistic models in R-code
    output of such model is save as .txt file into folder 'jupyter_notebook\R_script\R_log'
    these python scripts summarize these outputs files into one excel file 
"""

# get path to R_log files
path = get_path(file_name='', folder=PATH_R_LOG)
# list all txt files in the given path
file_list = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.txt')]


data_list = []
for file in file_list:
    data = parse_file(file)
    data_list.append(data)

df = pd.DataFrame(data_list)

d_info_all = []
for id, row in df.iterrows():
    if len(row['predictor']) > 1:
        continue
    file_name = row['file_name']
    parent_row_id = df.loc[df['file_name'] == file_name, 'predictor'].apply(len).idxmax()
    parent_row = df.loc[parent_row_id, :]
    pred = row['predictor'][0]
    pred_not_used = list(set(parent_row['predictor']) - set([pred]))

    df_coef = row['coefficients_df']
    df_coef_p = parent_row['coefficients_df']
    df_prob = row['predicted_probs_df'].apply(pd.to_numeric, errors='coerce')
    df_prob_p = parent_row['predicted_probs_df'].apply(pd.to_numeric, errors='coerce')
    d_info = {
        'formula': row['formula'],
        'response': row['response'],
        'predictor': row['predictor'][0],
        'AIC': row['AIC'],
        'p. AIC': parent_row['AIC'],
        'Null Deviance': row['Null Deviance'],
        'p. Null Deviance': parent_row['Null Deviance'],
        'Residual Deviance': row['Residual Deviance'],
        'p. Residual Deviance': parent_row['Residual Deviance'],
        'beta': df_coef.loc[df_coef['Variable'] == pred + '1', 'Estimate'].iloc[0],
        'p. beta': df_coef_p.loc[df_coef_p['Variable'] == pred + '1', 'Estimate'].iloc[0],
        'beta std': df_coef.loc[df_coef['Variable'] == pred + '1', 'Std. Error'].iloc[0],
        'p. beta std': df_coef_p.loc[df_coef_p['Variable'] == pred + '1', 'Std. Error'].iloc[0],
        'p-value': df_coef.loc[df_coef['Variable'] == pred + '1', 'Pr(>|z|)'].iloc[0],
        'p. p-value': df_coef_p.loc[df_coef_p['Variable'] == pred + '1', 'Pr(>|z|)'].iloc[0],
        'P[0]': df_prob.loc[df_prob[pred] == 0, 'predicted_probability'].iloc[0],
        'c[0]': df_prob.loc[df_prob[pred] == 0, 'count'].iloc[0],
        'P[1]': df_prob.loc[df_prob[pred] == 1, 'predicted_probability'].iloc[0],
        'c[1]': df_prob.loc[df_prob[pred] == 1, 'count'].iloc[0],
        'P[0, 0]': df_prob_p.loc[(df_prob_p[pred] == 0) & (df_prob_p[pred_not_used] == 0).all(axis=1), 'predicted_probability'].iloc[0],
        'c[0, 0]': df_prob_p.loc[(df_prob_p[pred] == 0) & (df_prob_p[pred_not_used] == 0).all(axis=1), 'count'].iloc[0],
        'P[0, 1]': df_prob_p.loc[(df_prob_p[pred] == 0) & (df_prob_p[pred_not_used] == 1).all(axis=1), 'predicted_probability'].iloc[0],
        'c[0, 1]': df_prob_p.loc[(df_prob_p[pred] == 0) & (df_prob_p[pred_not_used] == 1).all(axis=1), 'count'].iloc[0],
        'P[1, 0]': df_prob_p.loc[(df_prob_p[pred] == 1) & (df_prob_p[pred_not_used] == 0).all(axis=1), 'predicted_probability'].iloc[0],
        'c[1, 0]': df_prob_p.loc[(df_prob_p[pred] == 1) & (df_prob_p[pred_not_used] == 0).all(axis=1), 'count'].iloc[0],
        'P[1, 1]': df_prob_p.loc[(df_prob_p[pred] == 1) & (df_prob_p[pred_not_used] == 1).all(axis=1), 'predicted_probability'].iloc[0],
        'c[1, 1]': df_prob_p.loc[(df_prob_p[pred] == 1) & (df_prob_p[pred_not_used] == 1).all(axis=1), 'count'].iloc[0],
        'date': row['date'],
        'file_name': row['file_name'],
        'file_path': row['file_path'],
        'data_source_path': row['data_source_path']
    }
    d_info_all.append(d_info)

df_info = pd.DataFrame(d_info_all)
save_df(df_info, 'models_summary')

print('FINISHED')
