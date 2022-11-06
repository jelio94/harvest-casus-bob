"""File with several helper functions"""
import re
import pandas as pd


def load_csv(filename: str, separator=',', encoding='utf8') -> pd.DataFrame:
    """Simply loads the given csv file into a Pandas DataFrame"""
    return pd.read_csv(filename, sep=separator, encoding=encoding, skipinitialspace=True)


def preprocess_data_1(df: pd.DataFrame) -> pd.DataFrame:
    """Drops unnamed and ID columns, sets gemeentenaam and regio as multi-level index"""
    cleaned = df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'ID'], axis=1)
    # Remove number suffix from column names
    cleaned = cleaned.rename(lambda x: re.sub(r'_\d+', '', x), axis=1)
    # Strip additional whitespace from text rows
    cleaned['WijkenEnBuurten'] = cleaned['WijkenEnBuurten'].str.strip()
    cleaned['Gemeentenaam'] = cleaned['Gemeentenaam'].str.strip()
    cleaned['SoortRegio'] = cleaned['SoortRegio'].str.strip()
    cleaned['Codering'] = cleaned['Codering'].str.strip()
    # Remove data on Nederland, as it skews all normalization
    cleaned = cleaned[cleaned['Gemeentenaam'] != 'Nederland']
    # No data on buildable land available, unusable entries
    cleaned = cleaned.dropna(subset=['BouwgrondOpp'])
    cleaned = cleaned[cleaned['BouwgrondOpp'] > 0]
    # Set as multi-level index for easy grouping
    cleaned = cleaned.set_index(['Gemeentenaam', 'SoortRegio'], drop=True)
    return cleaned


def preprocess_data_2(df: pd.DataFrame) -> pd.DataFrame:
    """Drops unnamed columns, cleans labels"""
    cleaned = df.drop('Unnamed: 0', axis=1)
    cleaned = cleaned.rename(lambda x: re.sub(r'_\d+', '', x), axis=1)
    cleaned = cleaned.dropna(thresh=2).reindex()
    return cleaned


def sum_columns(df: pd.DataFrame, columns: [str], summed_name: str):
    """Sums all columns into summed_name. Handles missing values as 0"""
    df[summed_name] = df[columns].sum(axis=1)
    return df


def normalize_columns(df: pd.DataFrame, columns: [str], inverted=False):
    """Normalizes each column using min-max normalization, adds suffix 'Genormaliseerd' to column name"""
    for feature in columns:
        if inverted:
            df[f'{feature}GenormaliseerdInverted'] = 1 - (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
        else:
            df[f'{feature}Genormaliseerd'] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
    return df


def fill_missing(df: pd.DataFrame, columns: [str], fill_value: list):
    """Fills each column with the respective fill_value in list of same size"""
    return df.fillna(dict(zip(columns, fill_value)))


def count_zero_or_na(df: pd.DataFrame) -> pd.DataFrame:
    """Counts zeros, nans and combined per row"""
    df['Nul'] = (df == 0).sum(axis=1)
    df['Missend'] = df.isna().sum(axis=1)
    df['NulOfMissend'] = df['Nul'] + df['Missend']
    return df


def append_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combines features into new columns"""
    facility_columns = ['AfstandTotHuisartsenpraktijk', 'AfstandTotGroteSupermarkt', 'AfstandTotKinderdagverblijf',
                      'AfstandTotSchool']
    combined_facilities_name = 'GecombineerdeAfstandTotFaciliteiten'
    df = count_zero_or_na(df)
    # Filter based on criteria
    df = apply_criteria(df)
    # Fill missing values with max of that column
    df = fill_missing(df, facility_columns, df[facility_columns].max().tolist())
    df = sum_columns(df, facility_columns, combined_facilities_name)
    # Normalize columns, inverted for when a low score is better
    df = normalize_columns(df, [*facility_columns, 'BouwgrondOpp'])
    df = normalize_columns(df, [combined_facilities_name, 'Bevolkingsdichtheid'], inverted=True)
    return df


def apply_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows that do not meet criteria"""
    # More than one ha of building space is needed
    df = df[df['BouwgrondOpp'] > 1]
    # Rows with many zeros or nans result in a low score, but are really missing data
    df = df[df['Missend'] < 5]
    df = df[df['Nul'] < 5]
    return df


def make_score(df: pd.DataFrame, columns: [str], weights: [float]) -> pd.DataFrame:
    """Sums scoring features according to their weights"""
    df['Score'] = (df[columns] * weights).sum(axis=1)
    return df
