# %%
from classes.Dataset import DataSet
from classes.Pipeline import Pipeline
from extraction.mimic3_mean_100 import extract_features


import gc  # Import garbage collector interface

# from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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
            # "height_weight_switch",
            # "outlier_removal",
        ],
        "outlier_limits": "new_limits.json",
        "min_age": 18,
        "missing_rate": 0.5,
        "max_age": 90,
    },
    "ScalerEncoder": {},
    "Imputer": {
        "flag_missing": False,
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
            "IGANN",
            "LR",
            "DT",
            "EBM",
            "RF",
            "XGB",
            "PYGAM",
        ],
        "hpo_file": "hpo_grid_study.json",
        # "hpo_file": "default_hpo.json",
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
            # "MLP",
        ],
        "save_plots": True,
        "save_plot_data": True,
        "dist_plots": True,
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

targets = ["mortality", "LOS3", "LOS7"]

# %%
if pipe_params.get("Logging", None):
    update_logger(
        log, pipe_params["Logging"]["level"], pipe_params["Logging"]["file_mode"]
    )

log.info("Starting feature extraction...")
df = pd.read_csv(f"{DATA_DIR}/raw/MIMIC/mimic-complete.csv")

# create LOS targets
df.loc[:, "LOS3"] = df.loc[:, "LOS"] >= 3 * 24
df.loc[:, "LOS7"] = df.loc[:, "LOS"] >= 7 * 24

features = cat_features + num_features
df = df.filter(features + targets)
print(df.describe())
missing_rate = df.isna().mean()

# %% Runs for LOS3 and different impuations methods

log.info("Feature extraction completed.")

for target in [
    "mortality",
    "LOS3",
    "LOS7",
]:
    for imput_method in [
        "mean",
        "knn",
        "median",
        "iterative_LR",
        "iterative_RF",
    ]:

        log.info(
            f"######## Startet process for: {target}, and {imput_method} #########"
        )
        log.info(
            f"set addtional parameters for target:{target} and imputation method:{imput_method}"
        )
        # set parameters that variy
        dataset_params["Imputer"]["imputation_method"] = imput_method
        pipe_params["Memorizer"]["id"] = f"mimic_{target}_{imput_method}_test"
        pipe_params["Plotter"]["path"] = f"./plots/mimic_AoOR_{target}_{imput_method}"

        log.info("Creating dataset...")
        data = DataSet(
            name=f"mimic_mean_100%_{imput_method}_{target}",
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

        # clean up my mess to make space for more memory
        log.info(" Starting garbage collection")
        del data, pipe
        gc.collect()

        log.info("end of garbage collection")
