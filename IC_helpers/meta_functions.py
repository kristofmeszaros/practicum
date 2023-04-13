import os
import sys
import pandas as pd
import numpy as np
from IC_helpers import config
from IPython.display import display_html
import datetime
import time


def get_category_name_and_description(cat_id):
    name_value, description_value = 'missing', 'missing'
    # Iterate over all Category elements
    for category in config.category_root.iter('Category'):
        # Check if the ID attribute of the Category element is the one we're looking for
        if category.attrib['ID'] == str(cat_id):
            # Iterate over all Description and Name elements within the current Category element
            for child in category:
                if child.tag == 'Name':
                    # Extract the Value attribute from the Name element
                    name_value = child.attrib['Value']
                elif child.tag == 'Description':
                    # Extract the Value attribute from the Description element
                    description_value = child.attrib['Value']
            return name_value, description_value


def remove_all_null_column_pairs_and_unit_cols(df):
    """Remove columns with all null values and their corresponding columns with '.unit' suffix.
    The function will iterate through the columns of the input dataframe, check if the column without
    the '.unit' suffix has all null values and if it does:
    it will remove both the column with the suffix and the one without.
    Also removes unit columns.

    Parameters:
    df (pandas.DataFrame): the dataframe for which to remove the columns

    Returns:
    pandas.DataFrame: the input dataframe with the columns removed"""
    unit_cols = [col for col in df.columns if col.endswith('.unit')]

    # Iterate through the columns and check if the corresponding column without the suffix has all null values
    for col in unit_cols:
        base_col = col[:-5]  # remove the suffix, 5 length '.unit'
        if base_col in df.columns and df[base_col].isnull().all():
            print(f"removing {base_col} and the .unit")
            df.drop(base_col, axis='columns', inplace=True)

    df.drop(unit_cols, axis='columns', inplace=True)
    return df


def read_in_all_data(input_path, num_files=None):
    """
    This function takes a file path as input and read in `num_files` number of feather files in the folder
    (or all if `num_files` is not specified), performs some preprocessing and
    returns a dictionary of dataframes, where the key is the integer in the file name and the value is the dataframe
    """

    def remove_index_level_columns(df_):
        """
        Remove columns named 'index' and 'level_0' from a pandas dataframe.
        """
        df_.drop(["index"], inplace=True, axis=1)
        if 'level_0' in df_.columns:
            df_.drop(["level_0"], inplace=True, axis=1)
        return df_

    # Get a list of all .feather files in the folder
    feather_files = [f for f in os.listdir(input_path) if f.endswith('.feather')]
    df_dict = {}

    # If `num_files` is not specified, set it to the total number of files
    if not num_files:
        num_files = len(feather_files)

    # Read in `num_files` number of .feather files and add them to the dictionary
    for file in feather_files[:num_files]:
        key = file.split('_')[-1].split('.')[0]
        df = pd.read_feather(input_path + file)
        df = remove_all_null_column_pairs_and_unit_cols(df)
        df = remove_index_level_columns(df)
        df_dict[int(key)] = df
    return df_dict


def read_in_one_feather_file(input_path, category_number):
    def remove_index_level_columns(df_):
        """
        Remove columns named 'index' and 'level_0' from a pandas dataframe.
        """
        df_.drop(["index"], inplace=True, axis=1)
        if 'level_0' in df_.columns:
            df_.drop(["level_0"], inplace=True, axis=1)
        return df_

    df = pd.read_feather(f"{input_path}/frame_IceCat_Category_{category_number}.feather")
    df = remove_all_null_column_pairs_and_unit_cols(df)
    df = remove_index_level_columns(df)
    return df


def add_main_df_attributes(df_dict):
    """
    Extracts metadata for each dataframe in a dictionary and returns a dataframe containing the metadata
    for all input dataframes, sorted by the number of rows.

    Args:
    - df_dict (dict): A dictionary of dataframes, where each key is a category ID and the corresponding
                      value is a pandas DataFrame.

    Returns:
    - metadata_df (pandas DataFrame): A DataFrame containing metadata for each input DataFrame.
      The metadata includes:
        - id: The key of the DataFrame in the input dictionary
        - category: The value of the 'category_label' column of the first row of the DataFrame
        - rows: The number of rows in the DataFrame
        - columns: The number of columns in the DataFrame
        - metric_col_count: The number of metric columns
        - description: A description of the category, obtained using the `get_category_name_and_description` function.

    Example usage:
    ```
    metadata = add_main_df_attributes({'cat1': df1, 'cat2': df2})
    ```
    """
    have_data = []
    for cat_id, df in df_dict.items():
        cat_name, description = get_category_name_and_description(cat_id)
        unit_cols = [col for col in df.columns if "." in col]
        rowcount, colcount = df.shape[0], df.shape[1]
        metric_col_count = len(unit_cols)
        if colcount > 0: assert colcount - 4 == metric_col_count
        num_cols_with_30 = sum(df.count() >= 30)
        num_cols_with_50 = sum(df.count() >= 50)
        num_cols_with_100 = sum(df.count() >= 100)
        have_data.append([cat_id, cat_name, rowcount, metric_col_count, description,
                          num_cols_with_30, num_cols_with_50, num_cols_with_100])

    metadata_df = pd.DataFrame(have_data, columns=["id", "category", "rows", "feature_columns", "description",
                                                   "30_non_null", "50_non_null", "100_non_null"])
    metadata_df.sort_values(by=["rows"], ascending=False, inplace=True, ignore_index=True)
    return metadata_df


def get_column_metadata_for_category(cat_id):
    import xml.etree.ElementTree as ET
    feature_tree = ET.parse(config.feature_xml_filepath)
    feature_root = feature_tree.getroot()
    column_info = {}
    for category in feature_root.iter("Category"):
        if category.attrib["ID"] == str(cat_id):
            for feature in category.iter("Feature"):
                feature_id = feature.attrib["ID"]
                data = {
                    "name": feature.find("Name").attrib["Value"],
                    "unit": feature.find("Measure").attrib["Sign"],
                    "restricted_values": [rv.text for rv in feature.findall("RestrictedValue")]
                }
                column_info[int(feature_id)] = data
            return column_info
    print(f"cannot find category id {cat_id} in the file")


def print_first_x_dict_elements(dicti, x):
    count = 0
    for key, value in dicti.items():
        if count < x:
            print(key, value)
            count += 1
        else:
            break


def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)


def drop_some_columns(df, non_na_values_in_col_threshold):
    # drop columns which have too few non-null values, or only 1 unique non-null value,
    # or belong to the below meta columns
    print(f"df shape before removal{df.shape}")
    df.dropna(thresh=non_na_values_in_col_threshold, axis=1, inplace=True)
    for col in df.columns:
        # Count the number of unique non-null values in the column
        unique_values = df[col].nunique(dropna=True)
        is_it_ID_column = (all(df[col].value_counts() == 1) and
                           pd.api.types.is_numeric_dtype(df[col]) is False)
        # count the frequency of each item

        # If there is only one unique value (excluding nulls), drop the column
        if unique_values == 1 or is_it_ID_column is True \
                or col in ["id", "name", "category_id", "category_label"]:
            df.drop(col, axis=1, inplace=True)
    print(f"df shape after removal {df.shape}")
    numer_of_remaining_columns = df.shape[1]
    if numer_of_remaining_columns == 0:
        print("0 columns remaining, so exiting")
        sys.exit(0)
    return df


def get_numerical_and_sorted_cat_columns(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_values = df.isnull().sum()
    cat_cols = df.select_dtypes(include=[object]).columns.tolist()
    cat_cols = sorted(cat_cols, key=lambda col_: missing_values[col_])
    print(f"numerical columns: \n{num_cols}\n categorical columns: \n {cat_cols}")
    if len(num_cols) == 0 or len(cat_cols) == 0:
        print(f"{len(num_cols)} number of number columns, "
              f"{len(cat_cols)} number of cat columns, so exiting")
        sys.exit(0)
    return num_cols, cat_cols


def print_end_timings(start_time):
    now = datetime.datetime.now()
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Format the output as a string
    elapsed_time_string = "{:0>2.0f}:{:0>2.0f}:{:05.2f}".format(hours, minutes, seconds)
    print(f"FINISHED at:{now.strftime('%Y-%m-%d %H:%M:%S')}\n elapsed time: {elapsed_time_string}")
