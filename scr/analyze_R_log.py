import pandas as pd
import os

from supp.support_load import get_path
from supp.support_constants import PATH_R_LOG, PATH_DATA
from supp.support_parse_R_log import parse_file
import supp.support_load as sl
import argparse


"""
--------------------------------
    FILE BIO
--------------------------------
- file_name: analyze_R_log.py
- motivation:
    jupyter notebooks under 'jupyter_notebook\R_script' contain logistic models in R-code
    output of such model is save as .txt file into folder 'jupyter_notebook\R_script\R_log'
    these python scripts save coefficients into csv file
"""

# load argparse if any given
parser = argparse.ArgumentParser(description="Process a version string.")
parser.add_argument('--version', type=str, default='', help='Version string like "_v2"')
args = parser.parse_args()
version = args.version
#version = '_v2'
print(f"The version you entered is: '{version}'")

# get path to R_log files
path = get_path(file_name='', folder=PATH_R_LOG)
# list all txt files in the given path
files_names_list = [
    f'glm_vizier_vs_title_step{version}.txt',
    f'glm_vizier_vs_title_lasso{version}.txt',
    f'glm_vizier_vs_title_ridge{version}.txt'
]
file_list = [os.path.join(path, file) for file in files_names_list]

# Parse each R log file
data_list = []
for file in file_list:
    data = parse_file(file)
    data_list.append(data)
df = pd.DataFrame(data_list)

# Get coefficients map
model_summary = {row['file_name']: row['coefficients_df'] for id, row in df.iterrows()}
# Get model info (such AIC)
model_overview = {'model_overview': df.iloc[:, :-2].copy()}

df_name_map = sl.read_csv('df_title_name_r_map')
dict_name_map = df_name_map.set_index("key")["value"].to_dict()
for df in model_summary.values():
    df['Variable'] = df['Variable'].map(lambda x: dict_name_map.get(x, x))
    # remove 1 from factors names
    df.loc[df['Variable'] == 'father_was_vizier1', 'Variable'] = 'father_was_vizier'

# Define excell sheets
excel_sheets = {**model_overview, **model_summary}
# Save each DataFrame as a separate sheet in an Excel file
file_name_save = f'summary_model_coefficients_v0{version}.xlsx'
with pd.ExcelWriter(os.path.join(PATH_DATA, file_name_save), engine="openpyxl") as writer:
    for sheet_name, df in excel_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Excel file saved successfully!")
