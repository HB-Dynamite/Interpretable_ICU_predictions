# %%
import pandas as pd


def read_and_filter_data(file_path, identifiers, columns):
    """
    Reads a CSV file, filters rows based on the provided identifiers, and selects specific columns.

    Args:
    file_path (str): The path to the CSV file.
    identifiers (list): A list of identifiers to filter the DataFrame.
    columns (list): A list of columns to include in the resulting DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing the filtered and selected data.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    mask = df["ID"].apply(
        lambda ids: any(identifier in ids for identifier in identifiers)
    )
    # Filter the DataFrame based on the identifiers
    filtered_df = df[mask]
    # Select the specified columns
    result_df = filtered_df[columns]

    return result_df


# Example usage
file_path = "/home/lasse/Projects/TUDD-data-analysis/results/results_summary.csv"  # Replace with your CSV file path
identifiers = ["mimic"]  # Replace with your identifiers

# Specify the columns you want to see
cols = [
    # "Datetime",
    # "ModelConfigHash",
    # "LogfileName",
    # "ID",
    "name",
    "target",
    # "classification",
    # "cat_cols",
    # "num_cols",
    # "custom",
    # "Logging_level",
    # "Logging_file_mode",
    # "Trainer_input_split",
    # "Trainer_models",
    # "Trainer_hpo_file",
    # "Trainer_n_cv_folds",
    # "Plotter_models_to_plot",
    # "Plotter_save_plots",
    # "Plotter_save_plot_data",
    # "Plotter_dist_plots",
    # "Plotter_show_plots",
    # "Plotter_path",
    # "Memorizer_save",
    # "Memorizer_id",
    # "DataSetClassName",
    "model",
    # "AUROC",
    # "AUPRC",
    # "Accuracy",
    # "Precision",
    # "Recall",
    # "F1-Score",
    # "Confusion Matrix",
    # "params",
    "AUROC_CV_Mean",
    # "AUROC_CV_Std",
    # "AUPRC_CV_Mean",
    # "AUPRC_CV_Std",
    # "Accuracy_CV_Mean",
    # "Accuracy_CV_Std",
    # "Precision_CV_Mean",
    # "Precision_CV_Std",
    # "Recall_CV_Mean",
    # "Recall_CV_Std",
    # "F1-Score_CV_Mean",
    # "F1-Score_CV_Std",
]


result_df = read_and_filter_data(file_path, identifiers, cols)
print(result_df)

# %%

# %% custom code to extract imput method

result_df["imputation method"] = [name.split("_")[-2] for name in result_df["name"]]
result_df.drop(
    columns=["name"],
    inplace=True,
)
imput_methods = ["mean", "median", "knn", "RF", "LR"]

result_df = result_df[result_df["imputation method"].isin(imput_methods)]

# 1. Isolate rows where the imputation method is 'mean'
mean_imputation_df = result_df[result_df["imputation method"] == "mean"]


# 2. Create a reference table for AUROC values using mean imputation for each model-target combination
mean_auroc_ref = mean_imputation_df.set_index(["model", "target"])["AUROC_CV_Mean"]

# 3. Initialize a new column for AUROC difference
result_df["AUROC_delta_Mean"] = None

print(result_df)
# %%

# 4. Calculate the AUROC difference for each row compared to mean imputation
for index, row in result_df.iterrows():
    if row["imputation method"] != "mean":
        model_target_tuple = (row["model"], row["target"])
        # Check if the model-target combination exists in the index
        if model_target_tuple in mean_auroc_ref.index:
            # Directly access the single value using .loc
            mean_auroc = mean_auroc_ref.loc[model_target_tuple]
            # Ensure it's not returning a Series or another DataFrame
            if isinstance(mean_auroc, pd.Series):
                # If it's still a Series, take the first value
                mean_auroc = mean_auroc.iloc[0]
            # Calculate the difference and update the DataFrame
            result_df.at[index, "AUROC_delta_Mean"] = row["AUROC_CV_Mean"] - mean_auroc
        else:
            print(f"No entry found for {model_target_tuple}")


print(result_df)
# %%

# print(result_df)
for target, group in result_df.groupby("target"):

    sorted_group = group.sort_values(by=["model", "AUROC_CV_Mean"], ascending=False)
    sorted_group["AUROC_CV_Mean"] = sorted_group["AUROC_CV_Mean"] * 100
    sorted_group["AUROC_delta_Mean"] = sorted_group["AUROC_delta_Mean"] * 100

    # final_table = sorted_group[cols]
    print(
        sorted_group.to_latex(
            columns=["model", "imputation method", "AUROC_delta_Mean"],
            caption=(
                f"Comparison of AUROC Differences for Various Imputation Methods Relative to Mean Imputation for {target}",
                "",
            ),
            index=False,
            na_rep="--",
        )
    )

    print(sorted_group)


# %%
result_df.to_csv("test.csv")
# %%
