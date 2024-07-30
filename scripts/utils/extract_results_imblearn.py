# %%
import pandas as pd
import matplotlib.pyplot as plt

# use latex in labels
plt.rcParams["text.usetex"] = False


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
identifiers = [
    # "under_sampling",  # create plots for undersampling
    "over_sampling",  # create plots for oversampling
]  # Replace with other identifiers

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
    "AUROC_CV_Std",
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

result_df["sampling strategy"] = [name.split("_")[-3] for name in result_df["name"]]
result_df["ratio"] = [name.split("_")[-2] for name in result_df["name"]]
result_df.drop(
    columns=["name"],
    inplace=True,
)

print(result_df)
# %%
included_ratios = [
    "1",
    "0.8",
    "0.6",
    "0.4",
    "0.2",
]
# included_ratios = [
#     "1.0",
#     "0.9",
#     "0.8",
#     "0.7",
# ]

included_targets = [
    "mortality",
    # "LOS3",
    # "LOS7",
]

result_df = result_df[result_df["ratio"].isin(included_ratios)]
result_df = result_df[result_df["target"].isin(included_targets)]
result_df = result_df.drop(columns=["sampling strategy", "target"])

result_df.drop_duplicates(subset=["model", "ratio"], inplace=True)
print(result_df)
# %%

result_df = result_df.pivot(
    index="model",
    columns="ratio",
    # values=[
    #     "AUROC_CV_Mean",
    #     "AUROC_CV_Std",
    # ], # use this to include CV_std in plotting
    values="AUROC_CV_Mean",
)
print(result_df)
# %%

# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
# Define fixed color mappings
model_color_map = {
    "IGANN": "#56b4e9",  # Blue (Purple from the extracted colors)
    # "PYGAM": "#e69f00",  # Orange
    "GAM-Spline": "#e69f00",  # Orange
    "EBM": "#009e73",  # Green
    "XGB": "#d55e00",  # Red
    "RF": "#9467bd",  # Purple (Original value retained for RF)
    "LR": "#000000",  # Black
    "DT": "#cc79a7",  # Pink
}

# %%
plt.figure(figsize=(10, 6))
for model in result_df.index:
    plt.plot(
        result_df.columns,  # x-axis: ratios
        result_df.loc[model],  # y-axis: AUROC_CV_Mean values
        label=model,
        color=model_color_map[model],  # Use the predefined color for the model
    )
# plt.title("Mean CV AUROC Scores by undersampling Ratio and Model")
# plt.title("Mean CV AUROC Scores by oversampling Ratio and Model")

plt.xlabel(r"Ratio ( $\alpha$)", fontsize=14)
plt.ylabel("AUROC score")
plt.ylim((0.73, 0.865))
plt.grid(True)
# plt.legend(title="Model")
plt.show()


# Create a plot with CV std
# plt.figure(figsize=(10, 6))
# for model, color in zip(result_df.index, colors):
#     # Getting the means and standard deviations correctly for multi-level columns
#     means = result_df.loc[model, ("AUROC_CV_Mean", slice(None))]
#     stds = result_df.loc[model, ("AUROC_CV_Std", slice(None))]

#     # Ensure 'means' and 'stds' are series with 'ratios' as their index
#     means.index = included_ratios
#     stds.index = included_ratios

#     # Plotting the mean line
#     plt.plot(
#         included_ratios, means, label=model, color=color, lw=2
#     )  # 'lw' is line width

#     # Adding shaded area for standard deviations
#     plt.fill_between(
#         included_ratios, means - stds, means + stds, color=color, alpha=0.3
#     )  # 'alpha' for transparency

# # Customizing the plot
# plt.title("Mean CV AUROC Scores by Undersampling Ratio and Model with Std Deviation")
# plt.xlabel("Ratio")
# plt.ylabel("AUROC Score")
# plt.legend(title="Model")
# plt.grid(True)
# plt.show()
# %%
# create external legend
fig, ax = plt.subplots()
for model, color in model_color_map.items():
    ax.plot([], [], label=model, color=color, linewidth=10)
ax.legend(ncol=len(model_color_map))
plt.axis("off")  # Hide the axes
plt.show()
