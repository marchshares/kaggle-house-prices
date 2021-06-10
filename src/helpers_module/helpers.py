import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, MissingIndicator


def get_str_cols(df) -> list:
    """ Return list of column names which contains string data only"""

    return df.dtypes[df.dtypes == 'object'].index.values


def get_numeric_cols(df) -> list:
    """ Return list of column names which contains numeric data only"""

    return df.dtypes[df.dtypes != 'object'].index.values


def drop_str_cols(df) -> pd.DataFrame:
    """ Drop all strings columns and return DataFrame """

    return df.select_dtypes(exclude=['object'])


def print_cols_by_missing(df, n=20):
    """ Print n columns with percentage of missing values by descending """

    cols_by_missing = round(df.isna().sum() / df.shape[0] * 100, 2).sort_values(ascending=False)

    print("Row numbers: ", df.shape[0])
    print("Column name\t % of missing values")
    print(cols_by_missing.head(n))


def get_cols_by_missing(df, n=20):
    """ Get n columns with percentage of missing values by descending """

    cols_by_missing = round(df.isna().sum() / df.shape[0] * 100, 2).sort_values(ascending=False)

    return pd.DataFrame(cols_by_missing.head(n), columns=['% of missing values'])


def drop_cols_with_missing_more_threshold(df, threshold_of_missing) -> (pd.DataFrame, list):
    """ Drop all columns with percentage of missing values more then threshold_of_missing
        Return data and dropped columns names
    """

    missing_counts = df.isna().sum() / df.shape[0]
    columns_to_drop = df.columns[missing_counts > threshold_of_missing].values

    return df.drop(columns_to_drop, axis=1), columns_to_drop


def impute(df, strategy, fill_value=None) -> pd.DataFrame:
    """ Impute NaNs with strategy=['mean', 'median', 'most_frequent', 'constant']
        For 'constant' Impute fill_value in NaNs.
        If fill_value=None, impute 0 (for numeric) and 'missing_value' (for string)
    """

    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    data = imputer.fit_transform(df)

    return pd.DataFrame(data, columns=df.columns)


def encode_with_labels(df) -> pd.DataFrame:
    """ Encode with simple number labels """

    encoder = LabelEncoder()

    encoded_df = pd.DataFrame()
    for col in df.columns:
        encoded_df[col] = encoder.fit_transform(df[col])

    return encoded_df


def encode_with_labels_and_impute(df, strategy='mean') -> pd.DataFrame:
    """ Encode with simple labels and impute mean (of labels) in NaNs"""

    indicator = MissingIndicator(features='all')
    missing_indicator = indicator.fit_transform(df)

    df = impute(df, strategy='constant')  # impute dummy str NaNs
    df = encode_with_labels(df)

    # impute real np.nan back
    for i in range(0, df.shape[1]):
        missing_indicator_col = missing_indicator[:, i]
        df.iloc[missing_indicator_col, i] = np.nan

    return impute(df, strategy=strategy)


def encode_with_one_hot(dfs, n=10) -> pd.DataFrame:
    """ Encode with one hot encoding.
        The 1st df in dfs is provider for categories
        n - threshold for unique values in categories (if count > n just drop column)
        Will drop all original columns
    """

    main_df = dfs[0]
    one_hot_dfs = [pd.DataFrame() for i in range(len(dfs))]

    encoder = OneHotEncoder(sparse=False)

    str_columns = get_str_cols(main_df)
    for col in str_columns:
        if main_df[col].nunique() < n:
            encoder.fit(main_df[col].values.reshape(-1, 1))

            categories_count = len(encoder.categories_[0])
            col_names = [col + "_" + str(i) for i in range(categories_count)]

            for i in range(len(dfs)):
                df = dfs[i]
                one_hot_data = encoder.transform(df[col].values.reshape(-1, 1))

                one_hot_df = pd.DataFrame(one_hot_data, columns=col_names).astype('int')
                one_hot_dfs[i] = pd.concat([one_hot_dfs[i], one_hot_df], axis=1)

    for i in range(len(dfs)):
        dfs[i] = pd.concat([dfs[i], one_hot_dfs[i]], axis=1)
        dfs[i].drop(columns=str_columns, inplace=True)

    return dfs