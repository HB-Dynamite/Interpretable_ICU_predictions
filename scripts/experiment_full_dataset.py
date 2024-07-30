# %%
from classes.Dataset import DataSet
from classes.Pipeline import Pipeline

import gc  # Import garbage collector interface

from pprint import pprint
from utils.logger import log, update_logger
from utils.config import DATA_DIR
import logging
import pandas as pd


dataset_params = {
    "train_size": 0.8,
    "random_state": 1991,
    "Preprocessor": {
        "steps": [
            # "remove_minors",
            # "adapt_max_age",
            "remove_missing_data",
            # "outlier_removal",
        ],
        # option to remove outlieres
        # "outlier_limits": "new_limits.json",
        "min_age": 18,
        "missing_rate": 0.5,
        "max_age": 90,
    },
    "ScalerEncoder": {},
    "Imputer": {"flag_missing": False, "imputation_method": "mean"},
    # option to test for oversampling and undersampling
    "Balancer": {"oversampling_rate": None, "undersampling_rate": None},
}


pipe_params = {
    "Logging": {
        "level": logging.DEBUG,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        "file_mode": "w",  # choose "w" for overwrite, "a" for append
    },
    "Trainer": {
        "input_split": None,  # if None, the latest split after cleaning in the Dataset class will be selected TODO: better call trains split?
        "models": [
            "IGANN",
            "LR",
            "DT",
            "EBM",
            "RF",
            "XGB",
            "PYGAM",
        ],
        # full hpo will run a long time
        # "hpo_file": "hpo_grid_study.json",
        "hpo_file": "default_hpo.json",
        "n_cv_folds": 5,
    },
    "Evaluator": {},
    "Plotter": {
        "models_to_plot": [
            # "LR",
            # "DT",
            # "IGANN",
            # "EBM",
            # "PYGAM",
            # "RF",
            # "XGB",
        ],
        "save_plots": False,
        "save_plot_data": False,
        "dist_plots": False,
        "show_plots": False,
    },
    "Memorizer": {
        "save": True,
    },
}

# every model is trained in a seperated pipeline
models = [
    "LR",
    "DT",
    "EBM",
    "RF",
    "XGB",
    "IGANN",
    "PYGAM",
]

cat_features = [
    "Sex",
    "Eth",
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


# %%
if pipe_params.get("Logging", None):
    update_logger(
        log, pipe_params["Logging"]["level"], pipe_params["Logging"]["file_mode"]
    )

log.info("Starting feature extraction...")

df = pd.read_csv(f"{DATA_DIR}/raw/MIMIC/mimic-complete.csv")
# create LOS3 Target
df.loc[:, "LOS3"] = df.loc[:, "LOS"] >= 3 * 24
# create LOS7 Tagret
df.loc[:, "LOS7"] = df.loc[:, "LOS"] >= 7 * 24

# create list of num features to include

# drop features with high missingrate (this is also done in preprocessing)
missing_rate = df.isna().mean()
unvalid_features = list(missing_rate[missing_rate > 0.5].index)

# Filter columns for GCS subscales
gcs_columns_to_delete = [
    col for col in df.columns if "GCS" in col and "GCST" not in col
]

# also remove targets from the feature list
targets_to_drop = targets + ["LOS"]

# create list of all cols to remove (also cat features) added seperatly
columns_to_delete = (
    gcs_columns_to_delete + unvalid_features + targets_to_drop + cat_features
)
filtered_df = df.drop(columns=columns_to_delete)
num_features = filtered_df.columns


for target in targets_to_iterate:
    for model in models:
        log.info(
            f"######## Startet process for: {target} and all features, using model: {model} #########"
        )
        id = f"3_mimic_all_features_{target}_{model}"

        pipe_params["Memorizer"]["id"] = f"{id}"
        pipe_params["Plotter"]["path"] = f"./plots/mimic_AoOR_{id}"
        pipe_params["Trainer"]["models"] = [model]

        log.info("Creating dataset...")
        data = DataSet(
            name=f"{id}",
            target=target,
            classification=True,
            dataset_params=dataset_params,
            cat_cols=cat_features,
            num_cols=num_features,
            df=df,
            custom=False,
        )

        log.info("Dataset created successfully.")
        log.debug(f"Dataset Description: {data.describe}")

        log.info("starting pipeline")
        data.prepare_dataset()
        pipe = Pipeline(data, pipe_params=pipe_params)
        pipe.run()

        # clean up my mess to make space for more memory
        log.info(" Starting garbage collection")
        del data, pipe
        gc.collect()

        log.info("end of garbage collection")
