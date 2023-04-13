import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from IC_helpers.config import cv_results_path, aggregated_results_path, mice_verbose, max_mice_iter


def generate_dummy_data():
    # create a sample dataframe with missing values
    random_state = 123
    np.random.seed(random_state)
    n_rows = 100
    n_missing1 = 55
    n_missing2 = 60
    col1 = np.random.rand(n_rows)
    col2 = col1 * 2
    col3 = np.random.choice(['A', 'B', 'C', 'D'], size=n_rows)
    col4 = np.random.choice(['X', 'Y', 'Z'], size=n_rows)
    col5 = np.random.choice(['M', 'F'], size=n_rows)
    col1[n_missing1:] = None
    col3[2] = None
    col4[2] = None
    col5[n_missing2:] = None
    data = {'Numeric1': col1,
            'Numeric2': col2,
            'Category1': col3,
            'Category2': col4,
            'Category3': col5}
    data = {k: [None if x == 'N' else x for x in v] for k, v in data.items()}
    return pd.DataFrame(data)


def replace_20_percentage_of_non_nulls_for_each_column(df_):
    replaced_null_indices_dict = {}
    for col in df_.columns:
        ratio_to_replace = 0.2
        non_null_values = df_[col].dropna()
        n_replace = int(ratio_to_replace * len(non_null_values))
        replace_indices = np.random.choice(non_null_values.index, size=n_replace, replace=False)
        replaced_null_indices_dict[col] = replace_indices
        df_.loc[replace_indices, col] = None
    return df_, replaced_null_indices_dict


def mice_pipelines_and_paramgrids():
    class IdentityRegressor(BaseEstimator, RegressorMixin):
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return X

    # Define the hyperparameters to tune for the IterativeImputer
    imputer_param_grid = {
        'imputer__tol': [0.01],
        'imputer__initial_strategy': ['mean'],
        'imputer__imputation_order': ['ascending'],
        'imputer__n_nearest_features': [20],
    }

    # Define the hyperparameters to tune for each regressor
    rf_param_grid = {
        'imputer__estimator__n_estimators': [15, 30, 50],
        'imputer__estimator__max_depth': [5, 10, None],
        'imputer__estimator__min_samples_split': [5, 10, 20],
        'imputer__estimator__min_samples_leaf': [2, 4, 8]
    }

    knn_param_grid = {
        'imputer__estimator__n_neighbors': [3, 5, 7]
    }

    bayes_param_grid = {
        'imputer__estimator__alpha_1': [1e-6, 1e-2],
        'imputer__estimator__alpha_2': [1e-6, 1e-2],
        'imputer__estimator__lambda_1': [1e-6, 1e-2],
        'imputer__estimator__lambda_2': [1e-6, 1e-2]
    }

    svr_param_grid = {
        'imputer__estimator__kernel': ['linear', 'rbf'],
        'imputer__estimator__C': [0.1, 1, 10],
        'imputer__estimator__epsilon': [0.1, 0.2, 0.3],
    }

    en_param_grid = {
        'imputer__estimator__alpha': [0.1, 0.5, 1.0],
        'imputer__estimator__l1_ratio': [0.1, 0.5, 1.0],
    }

    # Combine all the param_grids into one dictionary
    param_grids = {
        'Random Forest': {**imputer_param_grid, **rf_param_grid},
        'KNN': {**imputer_param_grid, **knn_param_grid},
        'Bayesian Ridge': {**imputer_param_grid, **bayes_param_grid},
        'Mean': {},
        'Median': {},
        'Support Vector Regression': {**imputer_param_grid, **svr_param_grid},
        'Elastic Net': {**imputer_param_grid, **en_param_grid}
    }

    # Define the regressors
    rf = RandomForestRegressor(random_state=42)
    knn = KNeighborsRegressor()
    bayes = BayesianRidge()
    mean = DummyRegressor(strategy='mean')
    median = DummyRegressor(strategy='median')
    svr = SVR()
    en = ElasticNet()

    # Create a pipeline with an IterativeImputer and one of the regressors
    pipeline_rf = Pipeline([
        ('imputer', IterativeImputer(random_state=42, max_iter=max_mice_iter, estimator=rf, verbose=mice_verbose)),
        ('dummy', IdentityRegressor())
    ])

    pipeline_knn = Pipeline([
        ('imputer', IterativeImputer(random_state=42, max_iter=max_mice_iter, estimator=knn, verbose=mice_verbose)),
        ('dummy', IdentityRegressor())
    ])

    pipeline_bayes = Pipeline([
        ('imputer', IterativeImputer(random_state=42, max_iter=max_mice_iter, estimator=bayes, verbose=mice_verbose)),
        ('dummy', IdentityRegressor())
    ])

    pipeline_mean = Pipeline([
        ('imputer', IterativeImputer(random_state=42, max_iter=max_mice_iter, estimator=mean, verbose=mice_verbose)),
        ('dummy', IdentityRegressor())
    ])

    pipeline_median = Pipeline([
        ('imputer', IterativeImputer(random_state=42, max_iter=max_mice_iter, estimator=median, verbose=mice_verbose)),
        ('dummy', IdentityRegressor())
    ])

    pipeline_svr = Pipeline([
        ('imputer', IterativeImputer(random_state=42, max_iter=max_mice_iter, estimator=svr, verbose=mice_verbose)),
        ('dummy', IdentityRegressor())
    ])

    pipeline_en = Pipeline([
        ('imputer', IterativeImputer(random_state=42, max_iter=max_mice_iter, estimator=en, verbose=mice_verbose)),
        ('dummy', IdentityRegressor())
    ])

    # Combine all the pipelines into one dictionary
    pipelines = {
        'Random Forest': pipeline_rf,
        'KNN': pipeline_knn,
        'Bayesian Ridge': pipeline_bayes,
        'Mean': pipeline_mean,
        'Median': pipeline_median,
        'Support Vector Regression': pipeline_svr,
        'Elastic Net': pipeline_en
    }
    return pipelines, param_grids


def mice_negative_mse_of_all_cols(est, X_some_nulls, X_true):
    """
    negative since gridsearch wants to maximize the score, so if we want MSE be small, we want
     negative MSE to be big. Also the X_true can have actual missing values, so we fill them up with the imputed
     in this case to make their difference 0.
     The best estimator will maximize this score below anyway,
     it doesnt matter if it's not sound or hard to interpret..
    """
    X = est.predict(X_some_nulls)  # impute the dataframe
    mask = pd.isnull(X_true)
    # convert X to a pandas DataFrame
    X_df = pd.DataFrame(X, index=X_true.index, columns=X_true.columns)
    # assign values from X to missing values in X_true
    X_true[mask] = X_df[mask]

    # grid_Search is triying to MAXIMIZE the score. BIG MSE is actually bad, s
    # so that is why we take the negative, because we want to make the MSE as close to 0 as possible
    negative_mse = - np.average((X - X_true) ** 2)
    return negative_mse


def create_grid_searches(pipelines, param_grids, scoring):
    # Create a GridSearchCV object for each estimator with
    # the defined hyperparameters for both the estimator and the imputer
    grid_searches = {}
    for model_name, pipeline in pipelines.items():
        grid_search = GridSearchCV(pipeline, param_grid=param_grids[model_name], cv=3,
                                   scoring=scoring, refit=True, n_jobs=-1)
        grid_searches[model_name] = grid_search

    return grid_searches


def mice_fit_grid_searches(grid_searches, X_to_impute, X_true_values):
    print("\n[MICE FITTING STARTED]")
    for model_name, grid_search in grid_searches.items():
        print(f'[MICE] fitting model for {model_name}')
        grid_search.fit(X_to_impute, X_true_values)
    print("[MICE FITTING ENDED]\n")
    return grid_searches


def get_best_model_and_score_for_each_mice_column(best_estimator_for_each_pipeline, df_emptied, num_cols,
                                                  replaced_null_indices_dict, df_orig):
    imputed_dfs_by_best_estimators = {}
    for model_name, best_estimator in best_estimator_for_each_pipeline.items():
        model = best_estimator.named_steps['imputer']
        imputed_dfs_by_best_estimators[model_name] = pd.DataFrame(model.transform(df_emptied[num_cols]),
                                                                  columns=num_cols)

    scoring_results = {}
    for model_name, model_imputed_values in imputed_dfs_by_best_estimators.items():
        num_scoring_results = []
        for num_col in num_cols:
            imputed_values = model_imputed_values.loc[replaced_null_indices_dict[num_col], num_col]
            orig_values = df_orig.loc[replaced_null_indices_dict[num_col], num_col]
            rmse = np.sqrt(mean_squared_error(orig_values, imputed_values))
            corr_coef = orig_values.corr(imputed_values)
            mae = mean_absolute_error(orig_values, imputed_values)
            num_scoring_results.append([num_col, rmse, mae, corr_coef])

        num_scoring_results = pd.DataFrame(num_scoring_results, columns=['column', 'rmse', 'mae', 'correlation'])
        scoring_results[model_name] = num_scoring_results

    print(scoring_results)
    # create an empty dataframe to hold the results
    result_df = pd.DataFrame(columns=['column', 'rmse', 'mae', 'correlation', 'best_model'])
    for num_col in num_cols:
        # create an empty dictionary to hold the minimum rmse for each model
        min_rmse = {}

        # loop through each model
        for model_name, df in scoring_results.items():
            # find the row with the minimum rmse for this column and model
            row = df[df['column'] == num_col].sort_values(by=['rmse']).iloc[0]

            # add the minimum rmse to the dictionary
            min_rmse[model_name] = row['rmse']

        # find the model with the minimum rmse for this column
        best_model = min(min_rmse, key=min_rmse.get)

        # add the row to the result dataframe
        result_df = result_df.append({
            'column': num_col,
            'rmse': min_rmse[best_model],
            'mae': scoring_results[best_model][scoring_results[best_model]['column'] == num_col]['mae'].values[0],
            'correlation':
                scoring_results[best_model][scoring_results[best_model]['column'] == num_col]['correlation'].values[0],
            'best_model': best_model
        }, ignore_index=True)

    return result_df


def classification_fit_grid_searches(grid_searches, X, y):
    for model_name, grid_search in grid_searches.items():
        print(f'fitting model for {model_name}')
        grid_search.fit(X, y)
    return grid_searches


def evaluate_fitted_grid_searches_and_save_down_cv_results(grid_searches, name_first_part):
    best_estimators = {}
    mse_ = {}
    rank1_metrics = {}
    for model_name, grid_search in grid_searches.items():
        cv_result = pd.DataFrame(grid_searches[model_name].cv_results_). \
            sort_values(by="rank_test_score", ignore_index=True)
        rank1_metrics[model_name] = cv_result.loc[
            0, ["mean_fit_time", 'std_fit_time', 'mean_test_score', 'std_test_score']]
        cv_result.to_csv(f"{cv_results_path}/{name_first_part}_{model_name}_cvresults.csv", header=True, index=False)
        best_pipeline = grid_search.best_estimator_
        best_estimators[model_name] = best_pipeline
        mse_[model_name] = grid_search.best_score_

    return best_estimators, mse_, rank1_metrics


def imput_num_cols_after_mice_and_get_best_overall_model_name(mse_s, best_estimators, df_with_nulls, num_cols):
    best_model_name = max(mse_s, key=mse_s.get)
    best_model = best_estimators[best_model_name].named_steps['imputer']
    imputed_cols = pd.DataFrame(best_model.transform(df_with_nulls[num_cols]), columns=num_cols)
    return imputed_cols, best_model_name


def imput_cat_col(mse_s, best_estimators, df_with_nulls, num_cols):
    best_model_name = max(mse_s, key=mse_s.get)
    best_model = best_estimators[best_model_name].named_steps['model']
    return pd.DataFrame(best_model.transform(df_with_nulls[num_cols]), columns=num_cols)


def classification_pipelines_and_paramgrids():
    # Define the hyperparameters to tune for each classifier
    rf_param_grid = {
        'model__n_estimators': [15, 30, 50],
        'model__max_depth': [5, 10, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    knn_param_grid = {
        'model__n_neighbors': [3, 5, 7]
    }

    lr_param_grid = {
        'model__C': [0.1, 1, 10],
        'model__penalty': [None, 'l2']
    }

    svc_param_grid = {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    }

    nb_param_grid = {}

    # Combine all the param_grids into one dictionary
    param_grids = {
        'Random Forest': rf_param_grid,
        'KNN': knn_param_grid,
        'Logistic Regression': lr_param_grid,
        'Support Vector Classifier': svc_param_grid,
        'Naive Bayes': nb_param_grid,
        'Most Frequent': {}
    }

    # Define the classifiers
    rf = RandomForestClassifier(random_state=42)
    knn = KNeighborsClassifier()
    lr = LogisticRegression(random_state=42)
    freq = DummyClassifier(strategy='most_frequent')
    svc = SVC(random_state=42, probability=True)
    nb = GaussianNB()

    # Create a pipeline with one of the classifiers
    pipeline_rf = Pipeline([('model', rf)])
    pipeline_knn = Pipeline([('model', knn)])
    pipeline_lr = Pipeline([('model', lr)])
    pipeline_svc = Pipeline([('model', svc)])
    pipeline_nb = Pipeline([('model', nb)])
    pipeline_freq = Pipeline([('model', freq)])

    # Combine all the pipelines into one dictionary
    pipelines = {
        'Random Forest': pipeline_rf,
        'KNN': pipeline_knn,
        'Logistic Regression': pipeline_lr,
        'Naive Bayes': pipeline_nb,
        'Most Frequent': pipeline_freq,
        'Support Vector Classifier': pipeline_svc,
    }

    return pipelines, param_grids


def classification_calculate_scores(column_name, y_orig, y_pred, rank1_metrics, model_name):
    accuracy = round(accuracy_score(y_orig, y_pred), 3)
    precision = round(precision_score(y_orig, y_pred, average='weighted'), 3)
    recall = round(recall_score(y_orig, y_pred, average='weighted'), 3)
    f1 = round(f1_score(y_orig, y_pred, average='weighted'), 3)
    return [column_name, model_name, len(y_pred), accuracy, precision, recall, f1,
            rank1_metrics["mean_fit_time"], rank1_metrics['std_fit_time'],
            rank1_metrics['mean_test_score'], rank1_metrics['std_test_score']]


def save_down_all_results(name_pretag, overall_best_mice_df, mice_best_model_for_each_col_df,
                          best_categorical_results_df, categorical_all_results_df):
    overall_best_mice_df.to_csv(f"{aggregated_results_path}/{name_pretag}_mice_best_overall.csv",
                                header=True, index=False)
    mice_best_model_for_each_col_df.to_csv(f"{aggregated_results_path}/{name_pretag}_mice_best_for_each_col.csv",
                                           header=True, index=False)
    best_categorical_results_df.to_csv(f"{aggregated_results_path}/{name_pretag}_categorical_best_for_each_col.csv",
                                       header=True, index=False)
    categorical_all_results_df.to_csv(f"{aggregated_results_path}/{name_pretag}_categorical_all.csv",
                                      header=True, index=False)

    print("saved down the results!")


def mice_best_model_scoring_results(num_cols, df_emptied, replaced_null_indices_dict, df_orig, best_model_name):
    num_scoring_results = []
    for num_col in num_cols:
        imputed_values = df_emptied.loc[replaced_null_indices_dict[num_col], num_col]
        orig_values = df_orig.loc[replaced_null_indices_dict[num_col], num_col]
        rmse = np.sqrt(mean_squared_error(orig_values, imputed_values))
        corr_coef = orig_values.corr(imputed_values)
        mae = mean_absolute_error(orig_values, imputed_values)
        num_scoring_results.append([num_col, rmse, mae, corr_coef])
    overall_best_mice_df = pd.DataFrame(num_scoring_results, columns=['column', 'rmse', 'mae', 'correlation'])
    overall_best_mice_df["best_overall_model"] = best_model_name
    print(f"overall best performer mice df:\n {overall_best_mice_df}")
    return overall_best_mice_df


def scale_numerical_columns(df_emptied, df_orig, num_cols):
    scaler = StandardScaler()
    emptied = scaler.fit_transform(df_emptied[num_cols])
    orig = pd.DataFrame(scaler.transform(df_orig[num_cols]), columns=num_cols)
    return emptied, orig


def create_results_dfs(best_results, all_results):
    best_categorical_results_df = pd.DataFrame(
        best_results, columns=['column', 'best_model', 'test_length', 'accuracy', 'precision', 'recall', 'f1',
                               "mean_fit_time", 'std_fit_time', 'mean_test_score', 'std_test_score'])
    categorical_all_results_df = pd.DataFrame(
        all_results, columns=["column", "model", "mean_fit_time", 'std_fit_time', 'mean_test_score', 'std_test_score'])

    return best_categorical_results_df, categorical_all_results_df
