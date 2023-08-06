#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""world_analysis.py: analysis at the world level."""

import os
from typing import Union, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


array_like = Union[pd.Series, pd.DataFrame, List[float], np.ndarray]

# Parameters
RESULT_FOLDER = "results"
DATA_FOLDER = "data"
ID = "icu_id"
REGION = "WHO_Region"
TIME_COL = "quarter_patient_adm"
OUTCOME = "icu_dischrg_status"
AGE = "age"


def calc_avg(x: array_like) -> float:
    """Calcualte average percentage mortality rate.

    Args:
        x (array_like): outcome

    Returns:
        float: average percentage mortality rate
    """
    res = np.nanmean(100 * (1 - x))
    return res


def calc_std(x: array_like) -> float:
    """Calcualte std percentage mortality rate.

    Args:
        x (array_like): outcome

    Returns:
        float: std percentage mortality rate
    """
    res = np.nanstd(100 * (1 - x))
    return res


def calc_nb_nan(x: array_like) -> float:
    """Calcualte number of nan outcomes.

    Args:
        x (array_like): outcome

    Returns:
        float: number of nan outcomes
    """
    res = np.isnan(x).sum()
    return res


def get_color(i: int) -> str:
    """Get color from index.

    Args:
        i (int): index

    Returns:
        str: color hex code
    """
    dic_colors = {
        0: "#636EFA",
        1: "#EF553B",
        2: "#00CC96",
        3: "#AB63FA",
        4: "#FFA15A",
        5: "#19D3F3",
        6: "#FF6692",
        7: "#B6E880",
        8: "#FF97FF",
        9: "#FECB52",
    }
    return dic_colors.get(i % len(dic_colors), "#FFFFFF")


if __name__ == "__main__":
    # Load data
    df = pd.read_csv(os.path.join(DATA_FOLDER, "coredata1_2.csv"), low_memory=False)[
        [ID, TIME_COL, REGION, AGE]
    ]
    df = df[df[AGE] >= 18]  # keep only adults

    df = df[[ID, TIME_COL, REGION]]
    df[ID] = df[ID].astype("string")
    df2 = pd.read_csv(os.path.join(DATA_FOLDER, "coredata7.csv"), low_memory=False)[
        [ID, OUTCOME]
    ]
    df2[ID] = df2[ID].astype("string")
    df_merge = df.merge(df2, left_on=ID, right_on=ID, how="left")
    df_merge[ID] = df_merge[ID].astype("string")

    # Group by region and time
    res = df_merge.groupby(by=[REGION, TIME_COL]).agg(
        avg=pd.NamedAgg(column=OUTCOME, aggfunc=lambda x: calc_avg(x)),
        std=pd.NamedAgg(column=OUTCOME, aggfunc=lambda x: calc_std(x)),
        nb_nan=pd.NamedAgg(column=OUTCOME, aggfunc=lambda x: calc_nb_nan(x)),
    )

    # Plot average mortality rate with confidence interval
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    for reg_id, reg in enumerate(res.index.levels[0]):
        color = get_color(reg_id)
        ax.plot(res.loc[reg]["avg"], color=color, label=reg)
        ax.fill_between(
            list(res.loc[reg].index),
            (
                res.loc[reg]["avg"]
                - (1.96 / np.sqrt(res.loc[reg]["nb_nan"])) * res.loc[reg]["std"]
            ),
            (
                res.loc[reg]["avg"]
                + (1.96 / np.sqrt(res.loc[reg]["nb_nan"])) * res.loc[reg]["std"]
            ),
            color=color,
            alpha=0.1,
        )
        ax.set_xlabel("Quarter of patient admission")
        ax.set_ylabel("Avg Mortality rate (%)")

    # Plot number of patients
    ax = axs[1]
    counts = df_merge.groupby(by=[REGION, TIME_COL]).count()[ID]
    for reg_id, reg in enumerate(res.index.levels[0]):
        color = get_color(reg_id)
        ax.plot(counts.loc[reg], color=color, label=reg)
        ax.set_xlabel("Quarter of patient admission")
        ax.set_ylabel("Number of patients")

    fig.suptitle("Evolution of the average mortality rate by WHO region w/ 95% CI")
    fig.tight_layout()
    plt.legend(loc="best")
    fig.savefig(os.path.join(RESULT_FOLDER, "world_analysis.png"))
    plt.show()

    # Troncate US
    reg_id = 2
    reg = "Region of the Americas (AMR)"
    color = get_color(reg_id)
    time_periods = list(res.loc[reg].index)[:-1]
    plt.plot(time_periods, res.loc[reg]["avg"].to_numpy()[:-1], color=color)
    plt.fill_between(
        time_periods,
        (
            res.loc[reg]["avg"]
            - (1.96 / np.sqrt(res.loc[reg]["nb_nan"])) * res.loc[reg]["std"]
        ).to_numpy()[:-1],
        (
            res.loc[reg]["avg"]
            + (1.96 / np.sqrt(res.loc[reg]["nb_nan"])) * res.loc[reg]["std"]
        ).to_numpy()[:-1],
        color=color,
        alpha=0.1,
    )
    plt.xlabel("Quarter of patient admission")
    plt.ylabel("Avg Mortality rate (%)")
    plt.title("Evolution of the average mortality rate in the US w/ 95% CI")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FOLDER, "us_analysis.png"))
    plt.show()

    # Just numbers
    for reg_id, reg in enumerate(res.index.levels[0]):
        color = get_color(reg_id)
        plt.plot(counts.loc[reg], color=color, label=reg)
    plt.xlabel("Quarter of patient admission")
    plt.ylabel("Number of patients")
    plt.title("Number of datapoints per regions and quarter of patient admission")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_FOLDER, "world_count_analysis.png"))
    plt.show()
