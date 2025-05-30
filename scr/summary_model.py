import os
import numpy as np
import pandas as pd

import supp.support_load as sl
import supp.support_model_stats as sms
from supp.support_save import save_df

"""
 BIO: 
 This script summarizes all models.
 - Model are pre-defined in 'df_model_dict'.
 - It calculates statistics for each model.
 - It loads computed SHAP Absolute Mean values.
 - It loads model info.
 - It save results into csv file 'summary_model'.
"""


# Get model info
def get_model_info(df=None):
    if df is None:
        df = sl.read_csv('df_model_dict')
    # Required columns
    info_columns = ['order',
                    'model_name',
                    'feature_restriction',
                    'model_type',
                    'feature_version',
                    'file_name'
                    ]
    df_info = df.loc[:, info_columns].sort_values('order')
    df_info = df_info.sort_values('order')
    return df_info

if __name__ == '__main__':
    # ------------------------------
    # LOAD SETS AND MODELS
    # ------------------------------
    df_model = sl.read_csv('df_model_dict')
    df_model = df_model.sort_values('order')
    # get list of all model names
    model_name_list = df_model['model_name'].to_list()


    # ------------------------------
    # LOAD PREDICTED PROBABILITIES
    # ------------------------------
    df_summary_person = sl.read_csv('summary_person')
    df_probs_train = df_summary_person.loc[df_summary_person['set'] == 'Train', :]
    df_probs_val = df_summary_person.loc[df_summary_person['set'] == 'Validation', :]
    df_probs = df_summary_person.loc[df_summary_person['set'] == 'Test', :]
    df_probs_vizier = df_probs.loc[df_probs['vizier'] == 1, :]
    df_probs_non_vizier = df_probs.loc[df_probs['vizier'] == 0, :]


    # ------------------------------
    # LOAD SHAP VALUES
    # ------------------------------
    df_summary_title = sl.read_csv('summary_title')

    # ------------------------------
    # COMPUTE MODEL STATISTICS
    # ------------------------------
    model_summary_list = []
    for model_name in model_name_list:
        d = {
            'model_name': model_name,
            'mean': df_probs[model_name].mean(),
            'mean_vizier': df_probs_vizier[model_name].mean(),
            'mean_non_vizier': df_probs_non_vizier[model_name].mean(),
            'bottom_3th_vizier': df_probs_vizier[model_name].quantile(0.2, interpolation='lower'),
            'top_3th_non_vizier': df_probs_non_vizier[model_name].quantile(0.995, interpolation='higher'),
            'viziers_above_overlap': sms.find_viziers_above_overlap(df_probs, model_name),
            'viziers_in_overlap': sms.find_viziers_in_overlap(df_probs, model_name),
            'non_viziers_in_overlap': sms.find_non_viziers_in_overlap(df_probs, model_name),
            'persons_in_overlap': sms.find_persons_in_overlap(df_probs, model_name),
            'train_loss': sms.compute_bce_loss(df_probs_train['vizier'], df_probs_train[model_name]),
            'val_loss': sms.compute_bce_loss(df_probs_val['vizier'], df_probs_val[model_name]),
            'test_loss': sms.compute_bce_loss(df_probs['vizier'], df_probs[model_name]),
            'test_loss_vizier': sms.compute_bce_loss(df_probs_vizier['vizier'], df_probs_vizier[model_name]),
            'test_loss_non_vizier': sms.compute_bce_loss(df_probs_non_vizier['vizier'], df_probs_non_vizier[model_name]),
        }
        model_summary_list.append(d)

    # Convert dictionaries to dataframe
    df_model_stat = pd.DataFrame(model_summary_list)
    # Normalize stats
    df_model_stat_norm = sms.scale_df(df_model_stat)
    # Get SHAPs for each model
    df_model_shap = df_summary_title.set_index('title')
    df_model_shap = df_model_shap.transpose().reset_index(drop=False, names='model_name')
    # Get model info
    df_model_info = get_model_info(df_model)

    # Create summary model dataframe
    df_summary_model = pd.merge(df_model_info, df_model_stat, on='model_name', how='left')
    df_summary_model = pd.merge(df_summary_model, df_model_stat_norm, on='model_name', suffixes=('', '_norm'), how='left')
    df_summary_model = pd.merge(df_summary_model, df_model_shap, on='model_name', how='left')

    # Save summary file
    save_df(df_summary_model, 'summary_model')


    print('\n\nFINISHED')
