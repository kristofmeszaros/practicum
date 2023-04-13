import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import missingno as msno


def plot_missing_values(df, fig_width=10, fig_height=8, max_cols_per_plot=50):
    """
    Generates multiple horizontal bar charts showing the percentage of missing values in each column of the input
    dataframe. Breaks down columns into multiple plots if there are more than max_cols_per_plot columns in the input
    dataframe. The columns with the least percentage of missing values are plotted in the first chart.

    Args:
    df (pandas.DataFrame): The input dataframe.
    fig_width (float): The width of each plot in inches.
    fig_height (float): The height of each plot in inches.
    max_cols_per_plot (int): The maximum number of columns to show in each plot.

    Returns:
    None
    """
    total_rows = df.shape[0]
    missing_values_dict = {}
    for column in df.columns:
        column_name = '.'.join(column.split('.')[:-1]) if '.' in column else column
        missing_values = df[column].isna().sum()
        percentage_missing_values = round((missing_values / total_rows) * 100, 2)
        missing_values_dict[column_name] = percentage_missing_values

    # Convert dictionary to dataframe and sort by percentage in ascending order
    df_missing_values = pd.DataFrame.from_dict(missing_values_dict, orient='index', columns=['Percentage'])
    df_missing_values.sort_values('Percentage', ascending=True, inplace=True)

    # Generate multiple horizontal bar charts if there are more than max_cols_per_plot columns
    num_plots = (df.shape[1] // max_cols_per_plot) + 1
    for i in range(num_plots):
        start_col = i * max_cols_per_plot
        end_col = min((i + 1) * max_cols_per_plot, df.shape[1])
        df_missing_values_subset = df_missing_values.iloc[start_col:end_col, :]
        df_missing_values_subset = df_missing_values_subset.iloc[::-1]
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.barh(df_missing_values_subset.index, df_missing_values_subset['Percentage'])
        ax.set_xlabel('% Missing Values')
        ax.set_ylabel('Column')
        ax.set_title(f'Percentage of Missing Values in Columns {start_col + 1}-{end_col}')
        ax.set_xlim(0, 100)
        plt.show()


def missing_heatmap(df, rowstart, rowend, colstart, colend):
    # create a DataFrame indicating the presence or absence of missing values
    missing_df = df.iloc[rowstart:rowend, colstart:colend].isnull().transpose()

    # create a custom colormap that maps True to white and False to dark blue
    custom_cmap = ListedColormap(['#1E90FF', '#FFFFFF'])

    # create a heatmap of the missing value DataFrame using seaborn
    sns.heatmap(missing_df, cmap=custom_cmap, cbar=False)
    plt.title('Heatmap of Missing values', fontsize=20)  # title with fontsize 20
    plt.ylabel('feature', fontsize=15)  # x-axis label with fontsize 15
    plt.xlabel('index', fontsize=15)  # y-axis label with fontsize 15
    plt.show()


def draw_dendogram(df, colstart, colend):
    msno.dendrogram(df.iloc[:, colstart:colend], orientation='left')
    plt.title('Heatmap of Missing values', fontsize=20)  # title with fontsize 20
    plt.ylabel('feature', fontsize=15)  # x-axis label with fontsize 15
    plt.xlabel('index', fontsize=15)  # y-axis label with fontsize 15
    plt.show()
