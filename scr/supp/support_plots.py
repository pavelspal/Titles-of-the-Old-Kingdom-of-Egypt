import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# local
try:
    from supp.support_load import read_csv
    from supp.support_constants import PATH_IMG_SHAP
except:
    from support_load import read_csv
    from support_constants import PATH_IMG_SHAP

path_img = PATH_IMG_SHAP


def save_plot(file_name, folder=path_img):
    fig = plt.gcf()
    path = os.path.join(folder, file_name)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save figure
    fig.savefig(f"{path}.png", dpi=600, bbox_inches='tight')
    fig.savefig(f"{path}.pdf", bbox_inches='tight')
    print(f"Figure saved into {path}")

def get_model_info(model_name):
    feature_version = '1' if 'v1' in model_name else '2'
    if 'step' in model_name:
        model_type = 'step'
    elif 'lasso' in model_name:
        model_type = 'LASSO'
    else:
        model_type = 'Ridge'
    model = 'LR' if 'glm' in model_name else 'MLP'

    d = {'feature_version': feature_version,
         'model_type': model_type,
         'model': model}
    return d


def plot_shap_mean(s_shap_mean, model_name, data_set, category=1, plt_show=False, folder='shap_mean'):
    """
    Custom SHAP summary plot for binary features.
    Plots only SHAP values where corresponding feature values match `category`.

    :param s_shap_abs_mean: Pandas series with SHAP Mean values (same shape as `data`)
    :param model_name: Name of the model from which the s_shap_abs_mean are form
    :param category: Value (0 or 1) to filter features for plotting (default=1)
    """

    # Filter out nan values
    s = s_shap_mean.dropna()
    # Sort by SHAP importance
    s = s.sort_values(ascending=False)
    # Set limit for axis
    x_lim_left = min(s) * 1.07 if min(s) < 0 else 0
    x_lim_right = max(s) * 1.11

    # Plot
    base_width = 0.4
    num_bars = len(s)
    width = max(0, num_bars * base_width)
    plt.figure(figsize=(8, width))
    ax = sns.barplot(x=s.values, y=s.index,
                     hue=s, palette="coolwarm", legend=False,
                     zorder=2)
    # Draw only vertical line at x=0 to simulate partial grid
    #ax.axvline(0, color='gray', linestyle='-', linewidth=1, zorder=1)

    # Add text labels
    for index, value in enumerate(s):
        pos = max(value, 0)
        alignment = 'left' if value >= 0 else 'right'
        offset = 0
        ax.text(pos + offset, index, f"{value:.2f}", va='center', ha='left')
        ax.hlines(index, x_lim_left, 0, color='gray', linestyle='-',
                  linewidth=1, zorder=1, alpha=0.7)

    plt.xlabel("Mean SHAP Value")
    plt.ylabel("Feature")
    offset = 0.05
    plt.xlim(left=x_lim_left,
             right=x_lim_right)  # Ensure labels fit inside the figure
    d_info = get_model_info(model_name)
    plt.title(f"Custom SHAP Summary Plot\nModel: {d_info['model']}, Model type: {d_info['model_type']}, Feature version: {d_info['feature_version']}, Category: {category}, Set: {data_set}")

    file_name = f'SHAP_mean_{model_name}_category_{category}_set_{data_set}'
    save_plot(file_name, folder=os.path.join(path_img, folder))

    if plt_show:
        plt.show()
    else:
        plt.close()  # frees memory, especially in loops

    return True


def plot_shap_mean_concat(data_dict, model_name, category=1, plt_show=False, folder='merge_data_set'):
    """
    Custom SHAP summary plot for binary features.
    Plots multiple SHAP values where corresponding feature values match `category`, with each SHAP array having its own data.

    :param shap_data_pairs: Tuples of (SHAP values, corresponding DataFrame)
    :param category: Value (0 or 1) to filter features for plotting (default=1)
    """

    # Filter out nan values
    data_dict = {key: s.dropna().sort_index() for key, s in data_dict.items()}
    # Make list of all pandas series
    mean_abs_shap_list = list(data_dict.values())
    # Make list of all dictionary keys
    dataset_labels = list(data_dict.keys())

    # Sort by SHAP importance (average of all sets)
    mean_abs_avg = pd.DataFrame(mean_abs_shap_list).mean(axis=0)
    sorted_features = mean_abs_avg.sort_values(ascending=False).index.to_list()
    sorted_values_list = [s.loc[sorted_features].values for s in mean_abs_shap_list]

    # Create dataframe for plotting
    plot_data = pd.DataFrame({
        'Feature': np.tile(sorted_features, len(dataset_labels)),
        'SHAP Value': np.concatenate(sorted_values_list),
        'Dataset': np.repeat(dataset_labels, len(sorted_features))
    })

    # Plot
    plt.figure(figsize=(10, int(len(sorted_features)) / 2))
    ax = sns.barplot(x='SHAP Value', y='Feature', hue='Dataset', data=plot_data, palette="coolwarm")

    # Adjust text labels based on bar height
    bar_heights = [p.get_height() for p in ax.patches[:len(sorted_features)]]
    bar_spacing = max(bar_heights) * 1.0 if bar_heights else 0.2

    # Adjust text labels to avoid overlap
    x_lim_right = max(max(values[np.isfinite(values)]) for values in sorted_values_list)
    x_lim_left = min(min(values[np.isfinite(values)]) for values in sorted_values_list)
    x_lim_right = x_lim_right * 1.1
    x_lim_left = min(x_lim_left, 0) * 1.07
    max_height = 1e-10
    min_height = 1e10
    for i, sorted_values in enumerate(sorted_values_list):
        for index, value in enumerate(sorted_values):
            x_pos = max(value, 0)
            y_pos = index + (i - (len(sorted_values_list) - 1) / 2) * bar_spacing
            max_height = max(max_height, y_pos)
            min_height = min(min_height, y_pos)
            ax.text(x_pos, y_pos, f"{value:.2f}", va='center')
            ax.hlines(index, x_lim_left, min(value, 0), color='gray', linestyle='-',
                      linewidth=1, zorder=1, alpha=0.7)

    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.xlim(left=x_lim_left, right=x_lim_right)  # Ensure labels fit inside the figure
    plt.ylim(max_height + bar_spacing, min_height - bar_spacing)
    title = f"SHAP Summary Plot\nModel: {model_name}, Category: {category}"
    plt.title(title)
    plt.legend(title="Dataset", loc="lower right")

    file_name = f'SHAP_mean_{model_name}_category_{category}'
    save_plot(file_name, folder=os.path.join(path_img, folder))

    if plt_show:
        plt.show()
    else:
        plt.close()  # frees memory, especially in loops

    return True


def plot_shap_summary(data_dict, model_name, category=1, plt_show=False, folder='shap_summary'):
    # get feature order:
    first_key = list(data_dict.keys())[0]
    df_shap_first = data_dict[first_key]['shap']
    # Step 1: Compute mean absolute SHAP values for each feature
    mean_abs_shap = df_shap_first.abs().mean()
    # Step 2: Sort features by mean absolute SHAP value (descending)
    sorted_shap = mean_abs_shap.sort_values(ascending=False)
    # Step 3: Get the ordered list of feature names
    ordered_features = sorted_shap.index.tolist()

    explainer_list = []
    for key, d in data_dict.items():
        # Convert to shap.Explanation
        explainer = shap.Explanation(
            values=d['shap'][ordered_features].values,
            base_values=0,  # Dummy base_value unless you have it
            data=d['data'][ordered_features].values,
            feature_names=ordered_features
        )
        d['explainer'] = explainer

    for key, d in data_dict.items():
        plt.figure(figsize=(20, 10))
        shap.summary_plot(data_dict[key]['explainer'],
                          max_display=len(d['data'].columns),
                          show=plt_show
                          )
        # shap.summary_plot(
        #     d['shap'][ordered_features],
        #     features=d['data'][ordered_features],
        #     feature_names=ordered_features,
        #     max_display=len(d['data'].columns),
        #     show=plt_show
        # )
        plt.title(f"{key} SHAP Summary")
        # Save plot
        file_name = f'SHAP_SP_{model_name}_category_{category}_set_{key}'
        save_plot(file_name, folder=os.path.join(path_img, folder))

        if plt_show:
            plt.show()
        else:
            plt.close()  # frees memory, especially in loops

    return True
