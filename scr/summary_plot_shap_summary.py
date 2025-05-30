import os
import numpy as np
import pandas as pd

import supp.support_load as sl
from supp.support_plots import *
from summary_shap_calculator import load_data, get_shap_values_file_name


if __name__ == '__main__':
    # ------------------------------
    # LOAD SETS AND MODELS
    # ------------------------------
    df_model = sl.read_csv('df_model_dict')
    df_model = df_model.sort_values('order')

    for index, row in df_model.iterrows():
        model_name = row['model_name']
        version = row['feature_version']
        model_file_name = row['file_name']

        data_dict = {}
        for data_set in ['train', 'test']:
            df_data = load_data(data_set=data_set, version=version)
            file_name = get_shap_values_file_name(model_name, model_file_name, data_set)
            df_shap = sl.read_csv(file_name).set_index('ID_person')
            d = {
                'data': df_data[df_shap.columns],
                'shap': df_shap
            }
            data_dict[data_set] = d

        plot_shap_summary(data_dict, model_name)


    print('\n\nFINISHED')
