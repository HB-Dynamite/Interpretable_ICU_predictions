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
    "full_cv": True,
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
    "Imputer": {"flag_missing": False, "imputation_method": "mean"},
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
            "LR",
            "DT",
            "EBM",
            "IGANN",
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


cat_features = [
    "Sex",
    "Eth",
]

sensetive_num_features = ["Age"]

manual_num_features = [
    "Weight+100%mean",
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

auto_num_features_mortality = [
    "MBP+100%mean",
    "OS-10%mean",
    "RR+50%mean",
    "Ph+100%std",
    "GCST-10%mean",
    "Temp+100%mean",
    "DBP-50%mean",
    "GCST+50%std",
    "HR-25%max",
    "GLU-50%min",
    "Weight-10%min",
]

auto_num_features_LOS3 = [
    "Ph+50%std",
    "GCST+100%std",
    "DBP-10%min",
    "GCST-25%len",
    "RR+100%mean",
    "GCST-50%mean",
    "OS+100%min",
    "GCST-10%mean",
    "Ph-25%len",
    "SBP+100%min",
    "HR-10%min",
]


auto_num_features_LOS7 = [
    "GCST-25%len",
    "MBP+100%min",
    "OS-50%skew",
    "GCST+100%std",
    "HR-25%max",
    "Ph-25%len",
    "GCST-25%mean",
    "Ph+50%std",
    "OS+50%mean",
    "RR+100%mean",
    "RR+100%min",
]

targets = [
    "mortality",
    "LOS3",
    "LOS7",
]

featureset_dict = {
    "manual": {
        "sensetive": {
            "cat_features": cat_features,
            "num_features": manual_num_features + sensetive_num_features,
        },
        "not_sensetive": {
            "cat_features": [],
            "num_features": manual_num_features,
        },
    },
    "auto": {
        "mortality": {
            "sensetive": {
                "cat_features": cat_features,
                "num_features": auto_num_features_mortality + sensetive_num_features,
            },
            "not_sensetive": {
                "cat_features": [],
                "num_features": auto_num_features_mortality,
            },
        },
        "LOS3": {
            "sensetive": {
                "cat_features": cat_features,
                "num_features": auto_num_features_LOS3 + sensetive_num_features,
            },
            "not_sensetive": {
                "cat_features": [],
                "num_features": auto_num_features_LOS3,
            },
        },
        "LOS7": {
            "sensetive": {
                "cat_features": cat_features,
                "num_features": auto_num_features_LOS7 + sensetive_num_features,
            },
            "not_sensetive": {
                "cat_features": [],
                "num_features": auto_num_features_LOS7,
            },
        },
    },
}

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

missing_rate = df.isna().mean()
for target in targets:
    for sel in [
        "manual",
        "auto",
    ]:
        for sens in [
            "sensetive",
            "not_sensetive",
        ]:
            try:
                log.debug(
                    f"Starting iteration for target: {target}, selection: {sel}, sensitivity: {sens}"
                )

                # set IDs to extract results
                id = f"mimic_{target}_{sel}_selected_{sens}_2"

                if sel == "manual":
                    cat_features = featureset_dict[sel][sens]["cat_features"]
                    num_features = featureset_dict[sel][sens]["num_features"]
                else:
                    cat_features = featureset_dict[sel][target][sens]["cat_features"]
                    num_features = featureset_dict[sel][target][sens]["num_features"]

                log.info(
                    f"######## Started process for: {target}, selection: {sel}, sensitivity: {sens} #########"
                )
                log.info(f"using num_features: {num_features}")
                log.info(f"using cat_features: {cat_features}")

                pipe_params["Memorizer"]["id"] = f"{id}"
                pipe_params["Plotter"]["path"] = f"./plots/mimic_AoOR_{id}"

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

                log.info("Starting pipeline")
                data.prepare_dataset()
                pipe = Pipeline(data, pipe_params=pipe_params)
                pipe.run()

                # cleaning
                log.info("Starting garbage collection")
                del data, pipe
                gc.collect()
                log.info("End of garbage collection")

            except Exception as e:
                log.error(
                    f"Error in process for: {target}, selection: {sel}, sensitivity: {sens}"
                )
                log.error(str(e))
