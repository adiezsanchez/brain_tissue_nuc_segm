
import pandas as pd
import os 
import plotly.express as px


def classify_cells(df, method, min_max_per_marker, cell_populations):

    # Select all column names in 'final_df' that contain the substring method (i.e. 'avg_int')
    avg_int_columns = [col for col in df.columns if method in col]

    for marker_analysis in min_max_per_marker:

        marker = marker_analysis["marker"]
        min_max_avg_int = marker_analysis["min_max"]
        population = marker_analysis["population"]

        # Retrieve the column name from which the avg_int values should be read
        for column in avg_int_columns:
            if marker in column:
                column_name = column

        # Define if each nuclei label is positive (True) or negative (False) for a particular marker/population
        df[population] = (df[column_name] > min_max_avg_int[0]) & (df[column_name] < min_max_avg_int[1])

    # Define populations based on subpopulations
    for cell_population in cell_populations:
        # Extract population name and 
        cell_pop_name = cell_population["cell_pop"]
        subpopulations = cell_population["subpopulations"]
        
        # Initialize the column for this cell population and set all values to True
        df[cell_pop_name] = True

        # Loop through each subpopulation and its corresponding status (tuple)
        for subpop, status in subpopulations:
            # If the status is True, the cell should be positive for this subpopulation
            if status:
                # Perform a logical AND between the current column and the subpopulation column
                # This keeps only the rows where both conditions are True
                df[cell_pop_name] &= df[subpop]
            else:
                # If the status is False, the cell should be negative for this subpopulation
                # Negate the subpopulation column and perform a logical AND
                # This keeps only the rows where the subpopulation is False and the previous conditions are True
                df[cell_pop_name] &= ~df[subpop]

    return df

def calculate_perc_pops (results_path, method, min_max_per_marker, cell_populations):

    # Extract model_name and segmentation type from the results path

    model_name = results_path.parts[-1]
    segmentation_type = results_path.parts[-2]

    # Extract a list of filenames from the results path
    per_label_csvs = []

    for file_path in results_path.glob("*.csv"):
        if method in str(file_path) and "BP" not in str(file_path) and "SP" not in str(file_path):
            per_label_csvs.append(file_path)

    # Define the .csv path for the results
    csv_path = results_path / f"BP_populations_marker_+_per_label_{method}.csv"

    # List to store DataFrames before concatenating
    dfs_to_concatenate = []

    for csv in per_label_csvs:

        # Read the original per_label .csv
        df = pd.read_csv(csv, index_col=0)

        # Classify cells based on subpopulations
        df = classify_cells(df, method, min_max_per_marker, cell_populations)

        # Append to the list of DataFrames
        dfs_to_concatenate.append(df)

    # Concatenate all DataFrames
    final_df = pd.concat(dfs_to_concatenate, ignore_index=True)

    # Save the concatenated DataFrame to a .csv, overwriting any existing file
    final_df.to_csv(csv_path, index=False)

    # Read resulting Dataframe containing all the per_label info
    final_df = pd.read_csv(csv_path)

    # Identify cell population columns (those with boolean values)
    cell_pop_cols = final_df.select_dtypes(include=['bool']).columns

    # Group by filename and ROI
    grouped = final_df.groupby(['filename', 'ROI'])

    # Calculate the percentage of True values for each cell_pop column within each group
    def calc_percentage(x):
        max_label = x['label'].max()
        return (x[cell_pop_cols].sum() / max_label) * 100

    percentage_true = grouped.apply(calc_percentage, include_groups=False)

    # Reset index for better readability
    percentage_true = percentage_true.reset_index()

    # Save summary percentages for each population a .csv
    percentage_true.to_csv(path_or_buf= results_path / f"BP_populations_marker_+_summary_{method}.csv")

    return percentage_true, model_name, segmentation_type


def plot_perc_pop_per_filename_roi(df, model_name, segmentation_type):
    # List of columns to plot
    columns_to_plot = df.columns[2:]

    # Get unique ROIs
    unique_rois = df['ROI'].unique()

    # Iterate over each ROI
    for roi in unique_rois:
        df_roi = df[df['ROI'] == roi]
        for column in columns_to_plot:
            fig = px.scatter(df_roi, x='filename', y=column, hover_data={'index': df_roi.index})
            fig.update_layout(
                title={
                    'text': f'% of {column} in {roi}. Nuclei segm. model: {model_name}({segmentation_type})',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Filename",
                yaxis_title=f"Percentage of {column}",
                legend_title="ROI"
            )
            fig.show()