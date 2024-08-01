# AoOR Leveraging Interpretable Machine Learning in Intensive Care


## Overview
This repository supports the research paper titled "Leveraging Interpretable Machine Learning in Intensive Care," which explores the use of interpretable machine learning models in ICU settings to aid medical professionals in decision-making.

## Prerequisites

- Access to the MIMIC-III database via PhysioNet.
- Conda for managing the environment.

## Usage

### Step 1: Export MIMIC-III Data

1. Obtain access to the MIMIC-III database via [PhysioNet](https://physionet.org/).
2. Follow the instructions provided in the edited version of the MIMIC-III benchmark suite by Haraturiyan to export the data in CSV format.
   -(https://github.com/HB-Dynamite/mimic3-benchmarks_AoOR_data_export)
3. Save the exported data as mimic_complete.csv in the data/raw/MIMIC/ directory.

### Step 2: Set Up the Repository

1. Clone this repository into your workspace:
   
```sh
   git clone https://github.com/username/XXX.git
   cd TUDD-data-analysis
```
2. create and activate the conda environment 
```sh 
   conda env create -f env.yml
   conda activate envMIMIC
```
3. Run the Python scripts found in the scripts/ directory to execute the experiments. For example:
```sh 
   python scripts/experiment_X.py
```

## Structure: 
Interpretable_ICU_predictions/
├── data/
│   └── raw/
│       └── MIMIC/
│           └── mimic_complete.csv
├── hpo_configs/
├── logging/
├── output/
├── results/
├── scripts/
│   ├── classes/
│   ├── utils/
│   ├── experiment_X.py
│   ├── experiment_Y.py
│   └── ...
├── env.yml
├── README.md
└── ...
