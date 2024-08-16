# %%
from classes.Dataset import DataSet
from classes.Pipeline import Pipeline

import gc  # garbage collector

from pprint import pprint
from utils.logger import log, update_logger
from utils.config import DATA_DIR
import logging
import pandas as pd


dataset_params = {
    "train_size": 0.8,
    "random_state": 1991,
    "Preprocessor": {
        # optinal steps to preprocess the data
        "steps": [
            # "remove_minors",
            # "adapt_max_age",
            "remove_missing_data",
        ],
        "min_age": 18,
        "missing_rate": 0.5,
        "max_age": 90,
    },
    "ScalerEncoder": {},
    "Imputer": {"flag_missing": False, "imputation_method": "mean"},
    "Balancer": {"oversampling_rate": None, "undersampling_rate": None},
    "Selector": {
        "strategy": "SFFS",
        "k_features": 11,
        "forward": True,
        "floating": True,
        "scoring": "roc_auc",
        "cv": 5,
    },
}


cat_features = [
    # "Sex",
    # "Eth",
]

targets = [
    "mortality",
    "LOS3",
    "LOS7",
]

targets_to_iterate = [
    "mortality",
    "LOS3",
    "LOS7",
]


log.info("Starting feature extraction...")

df = pd.read_csv(f"{DATA_DIR}/raw/MIMIC/mimic-complete.csv")
# create LOS3 Target
df.loc[:, "LOS3"] = df.loc[:, "LOS"] >= 3 * 24
# create LOS7 Tagret
df.loc[:, "LOS7"] = df.loc[:, "LOS"] >= 7 * 24

# drop features with high missingrate (this is also done in preprocessing)
missing_rate = df.isna().mean()
unvalid_features = list(missing_rate[missing_rate > 0.5].index)

# filter columns for GCS subscales
gcs_columns_to_delete = [
    col for col in df.columns if "GCS" in col and "GCST" not in col
]

# remove targets from the feature list
targets_to_drop = targets + ["LOS"]

# create list of all cols to remove (also cat features) added seperatly
columns_to_delete = (
    gcs_columns_to_delete + unvalid_features + targets_to_drop + cat_features + ["Age"]
)
filtered_df = df.drop(columns=columns_to_delete)
num_features = filtered_df.columns


for target in targets_to_iterate:
    log.info(f"######## Startet feature selection for: {target} #########")
    id = f"mimic_feature_selection{target}"

    log.info("Creating dataset...")
    data = DataSet(
        name=f"{id}",
        target=target,
        classification=True,
        dataset_params=dataset_params,
        cat_cols=[],
        num_cols=num_features,
        df=df,
        custom=False,
    )

    log.info("Dataset created successfully.")
    log.debug(f"Dataset Description: {data.describe}")

    log.info("starting pipeline")
    data.prepare_dataset()
    data.select_features()

    # cleaning
    log.info("Starting garbage collection")
    del data
    gc.collect()
    log.info("End of garbage collection")
