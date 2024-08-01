# AoOR: Leveraging Interpretable Machine Learning in Intensive Care

## Overview

This repository supports the research paper "Leveraging Interpretable Machine Learning in Intensive Care," which explores the use of interpretable machine learning models in ICU settings to aid medical professionals in decision-making.

### Key Findings

Our research challenges the prevailing belief that only black-box models can provide high predictive performance in healthcare. We demonstrate that:

1. Interpretable models can achieve competitive predictive performance:
   - Only a minor decrease of 0.2-0.9 percentage points in AUROC compared to black-box models
   - Full interpretability maintained

2. Parsimonious models are highly effective:
   - Utilize only 2.2% of available patient features
   - Negligible performance drop relative to black-box models:
     - Range: 0.1 to 1.0 percentage points
     - Average: 0.5 percentage points

These findings aim to inspire further research and development of interpretable ML models in healthcare applications, showcasing that accuracy need not be sacrificed for interpretability.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{
  title={Leveraging Interpretable Machine Learning in Intensive Care},
  author={[Author Names]},
  journal={[Journal Name]},
  year={[Year]},
  volume={[Volume]},
  pages={[Pages]},
  doi={[DOI]}
}

## Prerequisites
- Python 3.8+
- Access to the MIMIC-III database via PhysioNet
- Conda for managing the environment

## Installation

### Step 1: Export MIMIC-III Data
1. Obtain access to the MIMIC-III database via [PhysioNet](https://physionet.org/).
2. Follow the instructions provided in the [edited version of the MIMIC-III benchmark suite](https://github.com/HB-Dynamite/mimic3-benchmarks_AoOR_data_export) to export the data in CSV format.
3. Save the exported data as `mimic_complete.csv` in the `data/raw/MIMIC/` directory.

### Step 2: Set Up the Repository
1. Clone this repository into your workspace:
   ```sh
   git clone https://github.com/username/XXX.git
   cd Interpretable_ICU_predictions
   ```
2. Create and activate the conda environment:
   ```sh
   conda env create -f env.yml
   conda activate envMIMIC
   ```

## Usage
Run the Python scripts found in the ``scripts/`` directory to execute the experiments. For example:
```sh
python scripts/experiment_X.py
```

## Experiments
For example one experiment could be: 
- ``experiment_X.py``: Mortality predicition
- ...

## Structure
```
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
```

## Contributing
We welcome contributions to this project. Please follow these steps:

1. Fork the repository
2. Create a new branch``(git checkout -b feature-branch)``
3. Make your changes and commit ``(git commit -am 'Add some feature')``
4. Push to the branch ``(git push origin feature-branch)``
5. Create a new Pull Request

## License
This project is licensed under the MIT License.

## Contact
For questions or feedback, please contact Lasse Bohlen at [lasse.bohlen@fau.de].

## Acknowledgments

### MIMIC-III database
```bibtex
@article{johnson2016mimic,
  title={MIMIC-III, a freely accessible critical care database},
  author={Johnson, Alistair EW and Pollard, Tom J and Shen, Lu and Lehman, Li-wei H and Feng, Mengling and Ghassemi, Mohammad and Moody, Benjamin and Szolovits, Peter and Anthony Celi, Leo and Mark, Roger G},
  journal={Scientific data},
  volume={3},
  number={1},
  pages={1--9},
  year={2016},
  publisher={Nature Publishing Group}
}
```
