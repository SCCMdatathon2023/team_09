#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tableone_.py: code to run a tableone."""

import os

import pandas as pd
from tableone import TableOne


FILE = "bq.csv"
DATA_FOLDER = "data"
RESULT_FOLDER = "results"
PRINT_TABLEONE = True
SAVE_TABLEONE = True


if __name__ == "__main__":
    # Load data
    df = pd.read_csv(os.path.join(DATA_FOLDER, FILE), low_memory=False)
    # TableOne
    columns = [
        "agegrp",
        "age",
        "sexcat",
        "bmi_category",
        "bmi_value",
        "racecat",
        "lack_resource",
        "hosp_status",
    ]
    categorical = [
        "agegrp",
        "sexcat",
        "bmi_category",
        "racecat",
        "lack_resource",
        "hosp_status",
    ]
    groupby = ["lack_resource"]
    table = TableOne(
        df,
        columns=columns,
        categorical=categorical,
        groupby=groupby,
        pval=True,
    )
    if PRINT_TABLEONE:
        print(table.tabulate(tablefmt="fancy_grid"))
    if SAVE_TABLEONE:
        table.to_excel(os.path.join(RESULT_FOLDER, "tableone.xlsx"))
