"""
TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
-----------------------------------------------------------------------
This module implements the classical TOPSIS method for multi-criteria
decision-making (MCDM).

Author: Nirjhar Gope
"""

import numpy as np
import pandas as pd


def normalize_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the decision matrix using vector normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Decision matrix with numerical criteria.

    Returns
    -------
    pd.DataFrame
        Normalized decision matrix.
    """
    denominator = np.sqrt((df ** 2).sum(axis=0))
    return df / denominator


def apply_weights(df: pd.DataFrame, weights: list) -> pd.DataFrame:
    """
    Apply weights to the normalized decision matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized decision matrix.
    weights : list
        List of weights corresponding to each criterion.

    Returns
    -------
    pd.DataFrame
        Weighted normalized decision matrix.
    """
    return df * np.array(weights)


def calculate_ideal_solutions(
    df: pd.DataFrame, impacts: list
) -> tuple[list, list]:
    """
    Calculate ideal best and ideal worst solutions.

    Parameters
    ----------
    df : pd.DataFrame
        Weighted normalized decision matrix.
    impacts : list
        Impact of each criterion ('+' for benefit, '-' for cost).

    Returns
    -------
    tuple
        Ideal best and ideal worst vectors.
    """
    ideal_best = []
    ideal_worst = []

    for i, impact in enumerate(impacts):
        column = df.iloc[:, i]
        if impact == '+':
            ideal_best.append(column.max())
            ideal_worst.append(column.min())
        else:
            ideal_best.append(column.min())
            ideal_worst.append(column.max())

    return ideal_best, ideal_worst


def calculate_distances(
    df: pd.DataFrame, ideal_best: list, ideal_worst: list
) -> tuple[pd.Series, pd.Series]:
    """
    Compute Euclidean distances from ideal best and worst solutions.

    Returns
    -------
    tuple
        Distance from ideal worst (S-) and ideal best (S+).
    """
    s_minus = np.sqrt(((df - ideal_worst) ** 2).sum(axis=1))
    s_plus = np.sqrt(((df - ideal_best) ** 2).sum(axis=1))
    return s_minus, s_plus


def calculate_topsis_score(
    s_minus: pd.Series, s_plus: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate TOPSIS performance score and ranking.

    Returns
    -------
    tuple
        TOPSIS score and rank.
    """
    score = s_minus / (s_minus + s_plus)
    rank = score.rank(ascending=False, method="dense").astype(int)
    return score, rank


def run_topsis(
    df: pd.DataFrame,
    criteria_cols: list,
    weights: list,
    impacts: list
) -> pd.DataFrame:
    """
    Execute the complete TOPSIS pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe containing candidate data.
    criteria_cols : list
        Column names used as decision criteria.
    weights : list
        Weights for each criterion.
    impacts : list
        Impact signs for each criterion.

    Returns
    -------
    pd.DataFrame
        DataFrame with TOPSIS score and rank.
    """
    decision_matrix = df[criteria_cols]

    normalized = normalize_matrix(decision_matrix)
    weighted = apply_weights(normalized, weights)

    ideal_best, ideal_worst = calculate_ideal_solutions(weighted, impacts)
    s_minus, s_plus = calculate_distances(weighted, ideal_best, ideal_worst)

    score, rank = calculate_topsis_score(s_minus, s_plus)

    result_df = df.copy()
    result_df["TopsisScore"] = score
    result_df["Rank"] = rank

    return result_df


weights = [0.2, 0.05, 0.15, 0.6]
impacts = ['+', '+', '+', '+']

criteria = ["Experience", "About", "Education", "Skill"]

final_df = run_topsis(
    df=dataframe,
    criteria_cols=criteria,
    weights=weights,
    impacts=impacts
)

print(final_df[["Name", "TopsisScore", "Rank"]])
