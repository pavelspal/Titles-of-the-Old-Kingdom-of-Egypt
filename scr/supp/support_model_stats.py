import numpy as np
import pandas as pd


def find_viziers_above_overlap(df, column):
    """
    Finds the index (iloc) of the first row in a sorted DataFrame (sorted by 'column' in descending order)
    where the column 'vizier' equals 0.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to sort by in descending order.

    Returns:
        int: The iloc of the first row where 'vizier' == 0, or iloc of last row if no such row is found.
    """
    # Take only Test set
    #test_df = df_model[df_model['set']=='Test']
    # Sort the dataframe by the specified column in descending order
    sorted_df = df.sort_values(by=[column, 'vizier'], ascending=False)
    # Reset index to start by 0
    sorted_df.reset_index(inplace=True)

    # Find the first row where 'vizier' == 0
    mask = (sorted_df['vizier'] == 0)
    if mask.any():  # Check if there are any True values in the mask
        return mask.idxmax()  # Returns the index of the first True
    else:
        return sorted_df.shape[0]  # Return df.shape[0] if no True is found


def find_non_viziers_in_overlap(df, column):
    """
    Finds number of 'non_viziers' that have higher proability then the worst predicted 'vizier'

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to sort by in descending order.

    Returns:
        int: Count of non_viziers that have higher perdicted probability then the worst predicted 'vizier'
    """
    # Take only Test set
    #test_df = df_test[df_test['set']=='Test']
    # Sort the dataframe by the specified column in descending order
    sorted_df = df.sort_values(by=[column, 'vizier'], ascending=False)
    # Reset index to start by 0
    sorted_df.reset_index(inplace=True)

    # Find the last index where vizier == 1
    last_index = sorted_df[sorted_df['vizier'] == 1].last_valid_index()

    # Compute non_viziers in overlap
    count = ((sorted_df['vizier']==0) & (sorted_df.index < last_index)).sum()

    return count


def find_viziers_in_overlap(df, column):
    """
    Finds number of 'viziers' that have lower proability then the best predicted 'non_vizier'

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to sort by in descending order.

    Returns:
        int: Count of viziers that have lower perdicted probability then the best predicted 'non_vizier'
    """
    # Take only Test set
    #test_df = df_test[df_test['set']=='Test']
    # Sort the dataframe by the specified column in descending order
    sorted_df = df.sort_values(by=[column, 'vizier'], ascending=False)
    # Reset index to start by 0
    sorted_df.reset_index(inplace=True)

    # Find the first index where vizier == 0
    last_index = sorted_df[sorted_df['vizier'] == 0].first_valid_index()

    # Compute non_viziers in overlap
    count = ((sorted_df['vizier']==1) & (sorted_df.index > last_index)).sum()

    return count


def find_persons_in_overlap(df, column):
    """
    Finds number of 'persons' that have probabilty between <best predicted non_vizier, worst predicted vizier>

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to sort by in descending order.

    Returns:
        int: Count of persons in overlap
    """
    return find_non_viziers_in_overlap(df, column) + find_viziers_in_overlap(df, column)


def compute_bce_loss(true_values, predicted_probs):
    """
    Computes the Binary Cross Entropy Loss (BCEWithLogitsLoss) given ground truth values
    and predicted probabilities as pandas.Series, ensuring stability by clamping probabilities.

    Parameters:
        true_values (pd.Series): Ground truth binary values (0 or 1).
        predicted_probs (pd.Series): Predicted probabilities (values in [0, 1]).

    Returns:
        float: The computed loss.
    """
    import torch
    import torch.nn as nn


    # Convert pandas.Series to PyTorch tensors
    true_values_tensor = torch.tensor(true_values.values, dtype=torch.float32)
    predicted_probs_tensor = torch.tensor(predicted_probs.values, dtype=torch.float32)

    # Clamp probabilities to avoid numerical issues
    predicted_probs_tensor = predicted_probs_tensor.clamp(min=1e-7, max=1 - 1e-7)

    # Define the loss criterion
    criterion = nn.BCELoss(reduction='mean')

    # Compute the loss
    loss = criterion(predicted_probs_tensor, true_values_tensor)

    return loss.item()


def scale_df(df):
    """
    Scales each float column of a DataFrame by its max value, leaving other columns unchanged.

    Parameters:
        df (pd.DataFrame): The input DataFrame with numeric and non-numeric columns.

    Returns:
        pd.DataFrame: A DataFrame where float columns are scaled by their median.
    """
    # Select float columns
    float_columns = df.select_dtypes(include=['float', 'int'])

    # Compute the median for each float column (explicitly setting axis=0)
    norm = float_columns.max(axis=0)

    # Scale only the float columns by their median
    scaled_floats = float_columns / norm

    # Replace the original float columns with their scaled versions
    scaled_df = df.copy()
    scaled_df[float_columns.columns] = scaled_floats

    return scaled_df