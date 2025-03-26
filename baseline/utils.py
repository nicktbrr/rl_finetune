from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def preprocess_data(train_df, test_df):
    """
    Preprocess the training and testing data.
    """
    # Remove unneeded columns
    list_drop = ['id', 'attack_cat'] if 'id' in train_df.columns else []
    if list_drop:
        train_df = train_df.drop(list_drop, axis=1)
        test_df = test_df.drop(list_drop, axis=1)

    # Apply data preprocessing steps
    train_df, test_df = clip_outliers(train_df, test_df)
    train_df, test_df = log_transform(train_df, test_df)
    train_df, test_df = encode_categorical(train_df, test_df)

    # Split features and target
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
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
    X_test = np.array(ct.transform(X_test))

    # Scale numerical features
    start_col = X_train.shape[1] - (X_train.shape[1] - len(cat_cols))
    sc = StandardScaler()
    X_train[:, start_col:] = sc.fit_transform(X_train[:, start_col:])
    X_test[:, start_col:] = sc.transform(X_test[:, start_col:])

    return X_train, y_train, X_test, y_test


def clip_outliers(train_df, test_df):
    """
    Clip extreme values to prevent outliers from affecting model training.
    """
    train = train_df.copy()
    test = test_df.copy()

    df_numeric = train.select_dtypes(include=[np.number])

    clip_values = {}

    for feature in df_numeric.columns:
        max_val = df_numeric[feature].max()
        median_val = df_numeric[feature].median()
        clip_threshold = df_numeric[feature].quantile(0.95)

        if max_val > 10 * median_val and max_val > 10:
            clip_values[feature] = clip_threshold
            train[feature] = np.where(
                train[feature] < clip_threshold, train[feature], clip_threshold)

    # Apply the same clipping to test data
    for feature, threshold in clip_values.items():
        test[feature] = np.where(
            test[feature] < threshold, test[feature], threshold)

    return train, test


def log_transform(train_df, test_df):
    """
    Apply log transformation to numerical features with many unique values.
    """
    train = train_df.copy()
    test = test_df.copy()

    df_numeric = train.select_dtypes(include=[np.number])

    transform_features = []

    for feature in df_numeric.columns:
        if df_numeric[feature].nunique() > 50:
            transform_features.append(feature)
            if df_numeric[feature].min() == 0:
                train[feature] = np.log(train[feature] + 1)
            else:
                train[feature] = np.log(train[feature])

    # Apply the same transformation to test data
    for feature in transform_features:
        if df_numeric[feature].min() == 0:
            test[feature] = np.log(test[feature] + 1)
        else:
            test[feature] = np.log(test[feature])

    return train, test


def encode_categorical(train_df, test_df):
    """
    Encode categorical features with many categories by keeping top categories.
    """
    train = train_df.copy()
    test = test_df.copy()

    df_cat = train.select_dtypes(exclude=[np.number])

    encode_features = {}

    for feature in df_cat.columns:
        if df_cat[feature].nunique() > 6:
            top_categories = train[feature].value_counts(
            ).head().index.tolist()
            encode_features[feature] = top_categories
            train[feature] = np.where(train[feature].isin(
                top_categories), train[feature], '-')

    # Apply the same transformation to test data
    for feature, top_categories in encode_features.items():
        test[feature] = np.where(test[feature].isin(
            top_categories), test[feature], '-')

    return train, test


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    plt.title(title)

    # Save the figure
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    return cm_path
