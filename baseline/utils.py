from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import mlflow
import json
import os


def preprocess_data(train_df, test_df=None, log_transformations=True, run_id=None):
    """
    Preprocess the training and testing data.
    """
    transformations = {}

    if run_id is not None and test_df is None:
        logged_transforms = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path='transformations.json')
        with open(logged_transforms, 'r') as f:
            transformations = json.load(f)
    # Remove unneeded columns
    list_drop = ['id', 'attack_cat'] if 'id' in train_df.columns else []
    if list_drop:
        train_df = train_df.drop(list_drop, axis=1)
        if test_df is not None:
            test_df = test_df.drop(list_drop, axis=1)

    # Apply data preprocessing steps
    train_df, test_df, transformations = clip_outliers(
        train_df, test_df, transformations)
    train_df, test_df, transformations = log_transform(
        train_df, test_df, transformations)
    train_df, test_df, transformations = encode_categorical(
        train_df, test_df, transformations)

    if log_transformations and test_df is not None:
        with open('transformations.json', 'w') as f:
            json.dump(transformations, f)
        mlflow.log_artifact('transformations.json')
        os.remove('transformations.json')

    # Split features and target
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    if test_df is not None:
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]

    # Identify categorical columns for one-hot encoding
    cat_cols = [i for i, col in enumerate(X_train.columns)
                if X_train[col].dtype == 'object']
    # Apply one-hot encoding and scaling
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), cat_cols)],
        remainder='passthrough')

    X_train = np.array(ct.fit_transform(X_train))

    if test_df is not None:
        X_test = np.array(ct.transform(X_test))

    # Scale numerical features
    start_col = X_train.shape[1] - (X_train.shape[1] - len(cat_cols))
    sc = StandardScaler()
    X_train[:, start_col:] = sc.fit_transform(X_train[:, start_col:])

    if test_df is not None:
        X_test[:, start_col:] = sc.transform(X_test[:, start_col:])
        return X_train, y_train, X_test, y_test

    return X_train, y_train


def clip_outliers(train_df, test_df=None, transformations={}):
    """
    Clip extreme values to prevent outliers from affecting model training.
    """
    train = train_df.copy()
    test = test_df.copy() if test_df is not None else None

    df_numeric = train.select_dtypes(include=[np.number])

    if "clip_values" not in transformations:
        clip_values = {}
        for feature in df_numeric.columns:
            max_val = df_numeric[feature].max()
            median_val = df_numeric[feature].median()
            clip_threshold = df_numeric[feature].quantile(0.95)

            if max_val > 10 * median_val and max_val > 10:
                clip_values[feature] = clip_threshold
                train[feature] = np.where(
                    train[feature] < clip_threshold, train[feature], clip_threshold)
        transformations["clip_values"] = clip_values
    else:
        clip_values = transformations["clip_values"]

    if test is not None:
        for feature, threshold in clip_values.items():
            test[feature] = np.where(
                test[feature] < threshold, test[feature], threshold)

    return train, test, transformations


def log_transform(train_df, test_df=None, transformations={}):
    """
    Apply log transformation to numerical features with many unique values.
    """
    train = train_df.copy()
    test = test_df.copy() if test_df is not None else None

    df_numeric = train.select_dtypes(include=[np.number])

    if "transform_features" not in transformations:
        transform_features = []
        for feature in df_numeric.columns:
            if df_numeric[feature].nunique() > 50:
                transform_features.append(feature)
                train[feature] = np.log(
                    train[feature] + 1) if df_numeric[feature].min() == 0 else np.log(train[feature])
        transformations["transform_features"] = transform_features
    else:
        transform_features = transformations["transform_features"]

    if test is not None:
        for feature in transform_features:
            test[feature] = np.log(
                test[feature] + 1) if df_numeric[feature].min() == 0 else np.log(test[feature])

    return train, test, transformations


def encode_categorical(train_df, test_df=None, transformations={}):
    """
    Encode categorical features with many categories by keeping top categories.
    """

    train = train_df.copy()
    test = test_df.copy() if test_df is not None else None

    df_cat = train.select_dtypes(exclude=[np.number])

    if "encode_features" not in transformations:
        encode_features = {}
        for feature in df_cat.columns:
            if df_cat[feature].nunique() > 6:
                top_categories = train[feature].value_counts(
                ).head().index.tolist()
                encode_features[feature] = top_categories
                train[feature] = np.where(train[feature].isin(
                    top_categories), train[feature], '-')
        transformations["encode_features"] = encode_features
    else:
        encode_features = transformations["encode_features"]

    if test is not None:
        for feature, top_categories in encode_features.items():
            test[feature] = np.where(test[feature].isin(
                top_categories), test[feature], '-')

    return train, test, transformations
