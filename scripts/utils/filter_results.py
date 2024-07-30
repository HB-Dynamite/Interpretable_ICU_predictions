import pandas as pd
from utils.config import RESULTS_DIR


def filter_results_by_id(id):
    """
    Filter the results CSV by the given id and save it as a new CSV file.

    Parameters:
    - id (str): The id to filter the results by.

    Returns:
    - None
    """
    results_path = RESULTS_DIR / "results_summary.csv"

    df = pd.read_csv(results_path)

    # filter the DataFrame by the given id
    df_filtered = df[df["ID"] == id]

    # save the filtered DataFrame as a new CSV file
    filtered_csv_file = RESULTS_DIR / f"results_{id}.csv"
    df_filtered.to_csv(filtered_csv_file, index=False)
