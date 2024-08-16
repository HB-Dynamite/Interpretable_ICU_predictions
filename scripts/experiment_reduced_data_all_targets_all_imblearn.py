# %%
from classes.Dataset import DataSet
from classes.Pipeline import Pipeline
from extraction.mimic3_mean_100 import extract_features


import gc  # garbage collector

from pprint import pprint
from utils.logger import log, update_logger
from utils.config import DATA_DIR
from utils.import_dataset import import_dataset_from_json
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
            # "height_weight_switch",
        ],
        "min_age": 18,
        "missing_rate": 0.5,
        "max_age": 90,
    },
    "ScalerEncoder": {},
    "Imputer": {
        "flag_missing": False,
        "imputation_method": "mean",
    },
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
            # "IGANN",
            # "LR",
            # "DT",
            # "EBM",
            # "RF",
            # "XGB",
            # "MLP",
            # "PYGAM",
        ],
        # full hpo will run a long time
        # "hpo_file": "hpo_grid_study.json",
        "hpo_file": "default_hpo.json",
        "n_cv_folds": 5,
    },
    "Evaluator": {},
    "Plotter": {
        # option to plot the feature effects and SHAP values
        "models_to_plot": [
            # "LR",
            # "IGANN",
            # "EBM",
            # "PYGAM",
            # "RF",
            # "XGB",
            # "MLP",
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


cat_features = [
    "Sex",
    "Eth",
]

num_features = [
    "Age",
    "Weight+100%mean",
    "Height+100%mean",
    "OS+100%mean",
    "Ph+100%mean",
    "Temp+100%mean",
    "RR+100%mean",
    "HR+100%mean",
    "GLU+100%mean",
    "DBP+100%mean",
    "SBP+100%mean",
    "MBP+100%mean",
    "GCST+100%mean",
]

targets = [
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
df.loc[:, "LOS7"] = df.loc[:, "LOS"] >= 7 * 24

features = cat_features + num_features
df = df.filter(features + targets)
print(df.describe())
missing_rate = df.isna().mean()

# %% Runs for LOS3 and different impuations methods

log.info("Feature extraction completed.")

for target in [
    "LOS7",
    "mortality",
    "LOS3",
]:
    for under_rate in [
        1,
        0.8,
        0.6,
        0.4,
        0.2,
        0.7,
        0.8,
        0.9,
        1.0,
    ]:

        log.info(f"######## Startet process for: {target}, and {under_rate} #########")
        log.info(
            f"set addtional parameters for target:{target} and imputation method:{under_rate}"
        )

        # set parameters that variy
        dataset_params["Balancer"]["undersampling_rate"] = under_rate

        pipe_params["Memorizer"]["id"] = f"mimic_{target}_under_sampling_{under_rate}_2"
        pipe_params["Plotter"][
            "path"
        ] = f"./plots/mimic_AoOR_{target}_undersampling_{under_rate}"

        log.info("Creating dataset...")
        data = DataSet(
            name=f"mimic_mean_100%_us_{under_rate}_{target}",
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

        data.prepare_dataset()
        pipe = Pipeline(data, pipe_params=pipe_params)
        pipe.run()

        # cleanig
        log.info(" Starting garbage collection")
        del data, pipe
        gc.collect()

        log.info("end of garbage collection")

# reset undersampling
dataset_params["Balancer"]["undersampling_rate"] = None

for target in [
    "LOS7",
    "mortality",
    # "LOS3",
]:
    for over_rate in [
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
    ]:

        log.info(f"######## Startet process for: {target}, and {over_rate} #########")
        log.info(
            f"set addtional parameters for target:{target} and imblearn method OS:{over_rate}"
        )

        # set parameters that variy
        dataset_params["Balancer"]["oversampling_rate"] = over_rate

        pipe_params["Memorizer"]["id"] = f"mimic_{target}_over_sampling_{over_rate}_2"
        pipe_params["Plotter"][
            "path"
        ] = f"./plots/mimic_AoOR_{target}_oversampling_{over_rate}"

        log.info("Creating dataset...")
        data = DataSet(
            name=f"mimic_mean_100%_os_{over_rate}_{target}",
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

        data.prepare_dataset()
        pipe = Pipeline(data, pipe_params=pipe_params)
        pipe.run()

        # cleaning
        log.info(" Starting garbage collection")
        del data, pipe
        gc.collect()

        log.info("end of garbage collection")
