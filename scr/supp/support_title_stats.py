import numpy as np
import pandas as pd


def conditional_probability(df, col1, col2):
    """
    Calculate the conditional probability P(col1=1 | col2=1) in a binary dataframe.

    Args:
    df (pd.DataFrame): Input DataFrame with binary values (0 or 1).
    col1 (str): The column for which probability is calculated (dependent variable).
    col2 (str): The column given as condition (independent variable).

    Returns:
    float: The conditional probability P(col1=1 | col2=1)
    """
    # If col1 or col2 is not valid column name, return nan
    if col1 not in df.columns or col2 not in df.columns:
        return np.nan

    # Count occurrences where both col1 and col2 are 1
    joint_count = ((df[col1] == 1) & (df[col2] == 1)).sum()

    # Count occurrences where col2 is 1
    col2_count = (df[col2] == 1).sum()

    # Compute conditional probability
    if col2_count == 0:
        return 0  # Avoid division by zero

    probability = joint_count / col2_count
    return round(probability, 4)


def get_probability(df, col1):
    """
    Calculate the probability P(col1=1) in a binary dataframe.

    Args:
    df (pd.DataFrame): Input DataFrame with binary values (0 or 1).
    col1 (str): The column for which probability is calculated (dependent variable).

    Returns:
    float: The conditional probability P(col1=1)
    """
    # If col1 is not valid column name, return nan
    if col1 not in df.columns:
        return np.nan

    # Count occurrences where col1 is 1
    joint_count = (df[col1] == 1).sum()

    # Count of all rows
    total_count = df.shape[0]

    # Compute probability
    if total_count == 0:
        return 0  # Avoid division by zero

    probability = joint_count / total_count
    return round(probability, 4)


def get_count(df, col1):
    """
    Calculate the probability P(col1=1) in a binary dataframe.

    Args:
    df (pd.DataFrame): Input DataFrame with binary values (0 or 1).
    col1 (str): The column for which probability is calculated (dependent variable).

    Returns:
    float: The conditional probability P(col1=1)
    """
    # If col1 or col2 is not valid column name, return nan
    if col1 not in df.columns:
        return np.nan

    # Count occurrences where both col1 and col2 are 1
    count = (df[col1] == 1).sum()

    return int(count)
