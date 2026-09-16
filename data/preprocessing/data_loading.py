import gc
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from pandas import DataFrame
from column_utils import raw_dtypes
import os


class DatasetVersion(Enum):
    V2 = "v2"
    V3 = "v3"

    def __str__(self):
        return self.value


class DatasetNames(Enum):
    BOT_IOT = "NF-BoT-IoT"
    CICIDS = "NF-CICIDS2018"
    TON_IOT = "NF-ToN-IoT"
    UNSW_NB15 = "NF-UNSW-NB15"

    def __str__(self):
        return self.name

    def get_full_name(self, version: DatasetVersion = DatasetVersion.V3):
        return f"{self.value}-{version.value}"


def validate_dataset_structure(df, dataset_name):
    """
    Validate and report on dataset structure, especially categorical columns.
    """
    print(f"Dataset structure validation for {dataset_name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")

    # Check for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        print(f"  Categorical columns found: {categorical_columns}")
        for col in categorical_columns:
            unique_vals = df[col].unique()
            print(
                f"    {col}: {len(unique_vals)} unique values - {list(unique_vals)[:10]}{'...' if len(unique_vals) > 10 else ''}")
    else:
        print(f"  No categorical columns found")

    # Check data types
    dtype_summary = df.dtypes.value_counts()
    print(f"  Data types: {dict(dtype_summary)}")

    return categorical_columns


def load_raw_dataset(dataset_name, version: DatasetVersion = DatasetVersion.V3, data_folder='./data'):
    full_name = dataset_name.get_full_name(version)
    df = pd.read_csv(os.path.join(data_folder, f"NetFlow_{version.value}_Datasets/{full_name}.csv", dtypes=raw_dtypes))

    # Validate structure and report categorical columns
    categorical_cols = validate_dataset_structure(df, full_name)

    return df


def load_samples(dataset_name, version: DatasetVersion = DatasetVersion.V3, data_folder='./data'):
    full_name = dataset_name.get_full_name(version)
    samples_path = os.path.join(data_folder, f"pickled sets/samples_{full_name}.pkl")

    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    with open(samples_path, "rb") as handle:
        samples = pickle.load(handle)

    return samples


def load_training_set(dataset_name, version: DatasetVersion = DatasetVersion.V3, data_folder='./data'):
    full_name = dataset_name.get_full_name(version)
    X_train = np.load(os.path.join(data_folder, f"pickled sets/X_train_{full_name}.pkl"), allow_pickle=True)
    return X_train


def load_validation_set(dataset_name, version: DatasetVersion = DatasetVersion.V3, data_folder='./data'):
    full_name = dataset_name.get_full_name(version)
    X_benign_validate = np.load(os.path.join(data_folder, f"pickled sets/X_benign_validate_{full_name}.pkl"),
                                allow_pickle=True)
    X_anomaly_validate = np.load(os.path.join(data_folder, f"pickled sets/X_anomaly_validate_{full_name}.pkl"),
                                 allow_pickle=True)
    return X_benign_validate, X_anomaly_validate


def load_test_set(dataset_name, version: DatasetVersion = DatasetVersion.V3, data_folder='./data'):
    full_name = dataset_name.get_full_name(version)
    X_benign_test = np.load(os.path.join(data_folder, f"pickled sets/X_benign_test_{full_name}.pkl"), allow_pickle=True)
    X_anomaly_test = np.load(os.path.join(data_folder, f"pickled sets/X_anomaly_test_{full_name}.pkl"),
                             allow_pickle=True)
    return X_benign_test, X_anomaly_test


@dataclass
class DatasetSamples:
    X_train_samples: list[DataFrame]
    X_benign_validate_samples: list[DataFrame]
    X_anomaly_validate_samples: list[DataFrame]
    X_benign_test_samples: list[DataFrame]
    X_anomaly_test_samples: list[DataFrame]


def generate_dataframe_samples(df: DataFrame, sample_size, n=1):
    samples = []
    for i in range(n):
        samples.append(df.sample(sample_size))

    return samples


def load_data_and_generate_dataset_samples(dataset, sample_size, benign_percentage=0.5, n=10,
                                           version: DatasetVersion = DatasetVersion.V3, data_folder='./data'):
    X_train = load_training_set(dataset, version, data_folder)
    X_benign_validate, X_anomaly_validate = load_validation_set(dataset, version, data_folder)
    X_benign_test, X_anomaly_test = load_test_set(dataset, version, data_folder)

    return generate_samples_from_dataframes(sample_size, benign_percentage, n, X_train, X_benign_validate,
                                            X_anomaly_validate, X_benign_test, X_anomaly_test)


def generate_samples_from_dataframes(sample_size, benign_percentage, n, X_train, X_benign_validate, X_anomaly_validate,
                                     X_benign_test, X_anomaly_test):
    benign_sample_size = int(sample_size * benign_percentage)
    anomaly_sample_size = sample_size - benign_sample_size

    X_train_samples = generate_dataframe_samples(X_train, sample_size, n)

    X_benign_validate_samples = generate_dataframe_samples(X_benign_validate, benign_sample_size, n)
    X_anomaly_validate_samples = generate_dataframe_samples(X_anomaly_validate, anomaly_sample_size, n)

    X_benign_test_samples = generate_dataframe_samples(X_benign_test, benign_sample_size, n)
    X_anomaly_test_samples = generate_dataframe_samples(X_anomaly_test, anomaly_sample_size, n)
    del X_train, X_benign_validate, X_benign_test, X_anomaly_validate, X_anomaly_test
    gc.collect()

    return DatasetSamples(X_train_samples,
                          X_benign_validate_samples,
                          X_anomaly_validate_samples,
                          X_benign_test_samples,
                          X_anomaly_test_samples)


def generate_samples_for_all_datasets(n=10, version: DatasetVersion = DatasetVersion.V3, data_folder='./data'):
    all_datasets_samples = {}
    for dataset_name in DatasetNames:
        all_datasets_samples[dataset_name] = load_data_and_generate_dataset_samples(dataset_name, sample_size=7062, n=n,
                                                                                    version=version,
                                                                                    data_folder=data_folder)
        gc.collect()
    return all_datasets_samples
