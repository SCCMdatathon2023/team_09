#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""create_virus.py: code to locally merge the virus dataset."""

import os

import pandas as pd
from tqdm import tqdm


# Parameters
ID = "icu_id"
DATA_FOLDER = "data"
DAY_COLUMN = "redcap_event_name"
TS_FILES = [f"coredata{i}" for i in range(4, 7)]
TS_COLUMNS = [f"oxygenation___{i}" for i in (1, 2, 3, 4, 5, 6, 9, 11)]
STATIC_COLUMNS = ["sex", "race", "bmi_value", "WHO_Region"]
if ID not in STATIC_COLUMNS:
    STATIC_COLUMNS = [ID] + STATIC_COLUMNS

if __name__ == "__main__":
    # Load static data
    df = pd.read_csv(os.path.join(DATA_FOLDER, "coredata1_2.csv"), low_memory=False)
    df[ID] = df[ID].astype("string")  # make sure the ID is a string
    df = df[STATIC_COLUMNS]  # keep only the static columns
    all_ids = list(df[ID].unique())  # get all the IDs
    all_days = [f"day{i}_arm_1" for i in range(29)]  # get all the days
    iterables = [all_ids, all_days]
    index = pd.MultiIndex.from_product(iterables)
    is_first_merge = True

    for i in tqdm(range(len(TS_FILES))):  # for each file, add the time varying features
        f = TS_FILES[i]
        new_df = pd.read_csv(
            os.path.join(DATA_FOLDER, f"{f}.csv"),
            low_memory=False,
        )
        new_df[ID] = new_df[ID].astype("string")

        common_columns = list(set(list(new_df.columns)).intersection(set(TS_COLUMNS)))
        if common_columns:  # if there are time varying features to add in the file
            grouped = new_df.groupby(by=[ID, DAY_COLUMN])[common_columns].max()
            completed_group = pd.DataFrame(grouped, index=index).fillna(
                0.0
            )  # fill the missing values with 0.0
            completed_group = completed_group.reset_index()
            if not (is_first_merge):
                completed_group.drop(columns=["level_1"], inplace=True)
            else:
                completed_group.rename(columns={"level_1": DAY_COLUMN}, inplace=True)

            if is_first_merge:
                df = df.merge(completed_group, left_on=ID, right_on="level_0").drop(
                    columns=["level_0"]
                )
                df[ID] = df[ID].astype("string")
                is_first_merge = False
            else:
                for col in list(completed_group.columns):
                    if col != "level_0":
                        df[col] = completed_group[col]

    # Additional preprocessing steps

    # Fill the missing values with 0.0
    df.fillna(0.0, inplace=True)
    # Merge some race categories
    race_mapping = {
        12: 6,
        8: 7,
        9: 7,
        "12": "6",
        "8": "7",
        "9": "7",
    }
    if "race" in STATIC_COLUMNS:
        df["race"] = df["race"].apply(lambda x: race_mapping.get(x, x))
    # Preprocess the date
    df[DAY_COLUMN] = df[DAY_COLUMN].apply(
        lambda x: int(x.split("day")[1].split("_")[0])
    )

    # Create a mapping dictionary for oxygenation values to scores for our custom score
    oxygenation_mapping = {2: 5, 3: 4, 1: 3, 5: 2, 6: 1, 9: 0}
    df["oxygenation___custom"] = df[
        [f"oxygenation___{i}" for i in oxygenation_mapping.keys()]
    ].apply(
        lambda row: oxygenation_mapping.get(
            int(row.idxmax().split("oxygenation___")[1]), 0
        ),
        axis=1,
    )

    # Save the file
    df.to_csv(os.path.join(DATA_FOLDER, "coredata.csv"), index=False)
