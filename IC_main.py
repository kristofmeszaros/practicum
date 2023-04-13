import sys
import datetime
import time
now = datetime.datetime.now()
print(f"START {now.strftime('%Y-%m-%d %H:%M:%S')}")
start_time = time.time()
import os
sys.path.append('/kristof.meszaros/pract')
os.chdir(os.path.dirname(__file__))
print(f"current working directory is: {os.getcwd()}")
from IC_helpers.meta_functions import read_in_one_feather_file, drop_some_columns, \
    get_numerical_and_sorted_cat_columns, print_end_timings
from IC_helpers.config import categories_xml_filepath, feather_path
from IC_helpers import config
import xml.etree.ElementTree as ET
from IC_helpers.machine_learning import replace_20_percentage_of_non_nulls_for_each_column, \
    mice_pipelines_and_paramgrids, mice_negative_mse_of_all_cols, \
    evaluate_fitted_grid_searches_and_save_down_cv_results, mice_fit_grid_searches, create_grid_searches, \
    imput_num_cols_after_mice_and_get_best_overall_model_name, classification_pipelines_and_paramgrids, \
    classification_fit_grid_searches, \
    classification_calculate_scores, get_best_model_and_score_for_each_mice_column, save_down_all_results, \
    mice_best_model_scoring_results, scale_numerical_columns, create_results_dfs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

pd.options.display.max_rows = 20
pd.options.display.max_columns = 30

category_tree = ET.parse(categories_xml_filepath)
config.category_root = category_tree.getroot()

category_number = int(sys.argv[1])
non_na_values_in_col_threshold = int(sys.argv[2])
print(f"category_number is {category_number}, threshold to drop is: {non_na_values_in_col_threshold}")
df = read_in_one_feather_file(feather_path, category_number)

############ preprocessing ##############################
df = drop_some_columns(df, non_na_values_in_col_threshold)
df_orig = df.copy()
# separate to num cols and sorted cat cols
num_cols, cat_cols = get_numerical_and_sorted_cat_columns(df)

# 2. For every single column randomly replace 20% of the non-null values with null
df_emptied = df_orig.copy()
df_emptied, replaced_null_indices_dict = replace_20_percentage_of_non_nulls_for_each_column(df_emptied)

################### MICE ##########################################
# 3. scale numerical columns based on test-set (with nulls put in previously), and test set same way...
df_emptied[num_cols], df_orig[num_cols] = scale_numerical_columns(df_emptied, df_orig, num_cols)
pipelines, param_grids = mice_pipelines_and_paramgrids()
# print(param_grids, pipelines, sep="\n")
grid_searches = create_grid_searches(pipelines, param_grids, scoring=mice_negative_mse_of_all_cols)

# Fit the GridSearchCV objects on the training data
X_to_impute = df_emptied[num_cols]
X_true_values = df_orig[num_cols]
fitted_grid_searches = mice_fit_grid_searches(grid_searches, X_to_impute, X_true_values)
# Get the best estimator from each GridSearchCV object, fit and get the best scores for each
best_estimator_for_each_pipeline, MSE_s, _ = evaluate_fitted_grid_searches_and_save_down_cv_results(
    fitted_grid_searches, name_first_part=f"cat{category_number}_threshold{non_na_values_in_col_threshold}_mice")

###  save down performance of each model for the cols
mice_best_model_for_each_col_df = get_best_model_and_score_for_each_mice_column(
    best_estimator_for_each_pipeline, df_emptied, num_cols, replaced_null_indices_dict, df_orig)
print(f"MICE best model for each col: \n {mice_best_model_for_each_col_df}")

### imput with the best of the estimators
df_emptied[num_cols], best_model_name = imput_num_cols_after_mice_and_get_best_overall_model_name(
    MSE_s, best_estimator_for_each_pipeline, df_emptied, num_cols)

### FINAL SCORES using the best model
overall_best_mice_df = mice_best_model_scoring_results(num_cols, df_emptied, replaced_null_indices_dict,
                                                       df_orig, best_model_name)
print("NUMERICAL columns finished. Starting categorical!")

############################### CATEGORICAL################
already_imputed_cat_cols = []
all_results = []
best_results = []
for cat_col in cat_cols:
    print(f"working on categorical column:{cat_col}")
    # Create a mask for the non-null values of the categorical column
    not_null_mask_for_training = df_emptied[cat_col].notnull()
    null_indices = replaced_null_indices_dict[cat_col]
    # Train a classifier on the non-null values of the categorical column
    X_train = df_emptied[num_cols + already_imputed_cat_cols][not_null_mask_for_training]
    # num cols won't have empty, but categorical can have after 1-hot encoding
    y_train = df_emptied[cat_col][not_null_mask_for_training]

    pipelines, param_grids = classification_pipelines_and_paramgrids()
    # print(param_grids, pipelines, sep="\n")
    grid_searches = create_grid_searches(pipelines, param_grids, scoring='accuracy')

    # Fit the GridSearchCV objects on the training data
    grid_searches = classification_fit_grid_searches(grid_searches, X_train, y_train)

    # Get the best estimator from each GridSearchCV object, fit and get the best scores for each
    cv_name_first_part = f"cat{category_number}_threshold{non_na_values_in_col_threshold}" \
                         f"_{cat_col.replace('/', '')}_classification"
    best_estimators, mse_s, rank1_metrics = evaluate_fitted_grid_searches_and_save_down_cv_results(
        grid_searches, name_first_part=cv_name_first_part)
    print(rank1_metrics)
    for model_name, metrics in rank1_metrics.items():
        one_result = [cat_col, model_name]
        for metric in metrics:
            one_result.append(metric)
        all_results.append(one_result)

    # calculate unbiased
    best_model_name = max(mse_s, key=mse_s.get)
    best_model = best_estimators[best_model_name].named_steps['model']
    X_test = df_emptied[num_cols + already_imputed_cat_cols].iloc[null_indices]
    y_test = best_model.predict(X_test)
    y_true = df_orig.loc[null_indices, cat_col]
    # print(y_test, best_model_name, y_true)
    best_results.append(classification_calculate_scores(
        cat_col, y_test, y_true, rank1_metrics[best_model_name], best_model_name))

    # replace the imputed one (y_train) + one with null_indices...
    # fill in the replaced nulls with the predicted ones
    df_emptied.loc[null_indices, cat_col] = y_test

    # Fit and transform the encoder on thise cat_col, and add it back to the df
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(df_emptied[[cat_col]]).toarray()
    encoded_df = pd.DataFrame(StandardScaler().fit_transform(encoded_features),
                              columns=encoder.get_feature_names_out([cat_col]))

    df_emptied.drop(cat_col, inplace=True, axis=1)
    df_emptied = pd.concat([df_emptied, encoded_df], axis=1)
    for col in encoder.get_feature_names_out([cat_col]):
        already_imputed_cat_cols.append(col)
    # print(encoder.get_feature_names_out([cat_col]))

best_categorical_results_df, categorical_all_results_df = create_results_dfs(best_results, all_results)

########### Save down all results
save_down_all_results(f"cat{category_number}_threshold{non_na_values_in_col_threshold}",
                      overall_best_mice_df, mice_best_model_for_each_col_df,
                      best_categorical_results_df, categorical_all_results_df)

print_end_timings(start_time)
