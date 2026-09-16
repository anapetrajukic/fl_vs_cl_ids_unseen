import gc
import pickle
from typing import Union, Callable
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import os
import numpy as np

# import psutil

from data_loading import (
    DatasetNames,
    DatasetVersion,
    generate_samples_from_dataframes,
    load_raw_dataset,
)
from preprocessing import (
    remove_attack_column,
    remove_network_specific_features,
    extract_label,
    divide_into_benign_and_anomaly,
    remove_inf_values,
    expand_tcp_flags,
    remove_duplicate_rows,
    remove_nan_values,
    print_anomaly_stats,
    analyze_duplicates,
)


def generate_cleaned_train_val_and_test_set(
        dataset: Union[DatasetNames, DataFrame],
        set_fractions: tuple[float, float, float] = (0.5, 0.5, 0.5),
        version: DatasetVersion = DatasetVersion.V3,
):
    if isinstance(dataset, DatasetNames):
        df = load_raw_dataset(dataset, version)
    else:
        df = dataset

    print(f"Initial dataset shape: {df.shape}")
    print(f"Initial sample rows:\n{df.head(3)}")
    print()

    print_anomaly_stats(df, "initial loading")
    print()
    initial_rows = len(df)

    # Print data types of all features
    print("Data types of all features:")
    feature_types = df.dtypes
    type_counts = feature_types.value_counts()
    print(f"Type summary: {dict(type_counts)}")
    print("Individual feature types:")
    for col, dtype in feature_types.items():
        print(f"  {col}: {dtype}")
    print()

    df = remove_network_specific_features(df)
    print_anomaly_stats(df, "network-specific feature removal")
    print()

    df = df.drop("MIN_IP_PKT_LEN", axis=1)
    df = df.drop("MAX_IP_PKT_LEN", axis=1)

    df = remove_nan_values(df)
    print_anomaly_stats(df, "NaN removal")
    print()

    df = remove_inf_values(df)
    print_anomaly_stats(df, "infinite values removal")
    print()

    # Remove Attack column before label extraction (it's categorical and not needed)
    remove_attack_column(df)
    print_anomaly_stats(df, "attack column removal")
    print()

    df = expand_tcp_flags(df)
    print_anomaly_stats(df, "TCP flags expansion")
    print()

    # Calculate skewness for each column
    skewness = df.skew()
    print("Skewness of features:")
    print(skewness[abs(skewness) > 1].sort_values(ascending=False))
    print()

    # Identify highly skewed features and apply log transformation
    highly_skewed_features = skewness[abs(skewness) > 1].index.tolist()
    if highly_skewed_features:
        print(f"Applying log transformation to {len(highly_skewed_features)} highly skewed features:")
        print(f"Features: {highly_skewed_features}")

        for feature in highly_skewed_features:
            # Add small constant to handle zero and negative values
            min_val = df[feature].min()
            if min_val <= 0:
                shift_constant = abs(min_val) + 1
                df[feature] = df[feature] + shift_constant
                print(f"  {feature}: shifted by {shift_constant} before log transform")

            # Apply log transformation
            df[feature] = np.log1p(df[feature])  # log1p is log(1+x), more stable for small values

        # Recalculate skewness after transformation
        new_skewness = df[highly_skewed_features].skew()
        print("Skewness after log transformation:")
        print(new_skewness.sort_values(ascending=False))
        print()
    else:
        print("No highly skewed features found (threshold: |skewness| > 1)")
        print()

    final_rows = len(df)
    removed_rows = initial_rows - final_rows
    reduction_percentage = removed_rows / initial_rows * 100
    print(
        f"Total preprocessing: {initial_rows} -> {final_rows} rows "
        f"({removed_rows} removed, {reduction_percentage:.2f}% reduction)"
    )
    print(f"Final preprocessed shape: {df.shape}")
    print(f"Final sample rows:\n{df.head(3)}")
    print()

    # Extract labels (handles categorical to numeric conversion)
    X, y = extract_label(df)
    print(f"Features extracted: {X.shape[1]} features, {len(y)} samples")
    print(f"Feature sample:\n{X.head(3)}")
    print()
    del df
    gc.collect()

    X_anomaly, X_benign = divide_into_benign_and_anomaly(X, y)
    print(f"Benign sample:\n{X_benign.head(3)}")
    print(f"Anomaly sample:\n{X_anomaly.head(3)}")
    print()
    del X
    gc.collect()

    X_train, X_benign_val_and_test = train_test_split(
        X_benign, train_size=set_fractions[0]
    )
    X_benign_validate, X_benign_test = train_test_split(
        X_benign_val_and_test, train_size=set_fractions[1]
    )
    X_anomaly_validate, X_anomaly_test = train_test_split(
        X_anomaly, train_size=set_fractions[2]
    )

    print("Final dataset splits:")
    print(f"  Training (benign only): {X_train.shape}")
    print(
        f"  Validation: {X_benign_validate.shape} benign + {X_anomaly_validate.shape} anomaly"
    )
    print(f"  Test: {X_benign_test.shape} benign + {X_anomaly_test.shape} anomaly")
    print(f"Training sample:\n{X_train.head(3)}")
    print()

    print("Experiments using the following columns: ")
    for col in X_train.columns:
        print(col)

    return X_train, X_benign_validate, X_anomaly_validate, X_benign_test, X_anomaly_test


def main():
    # You can change this to DatasetVersion.V2 to process V2 datasets
    dataset_version = DatasetVersion.V3

    for dataset_name in DatasetNames:
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset_name.get_full_name(dataset_version)}")
        print(f"{'=' * 60}")

        (
            X_train,
            X_benign_validate,
            X_anomaly_validate,
            X_benign_test,
            X_anomaly_test,
        ) = generate_cleaned_train_val_and_test_set(
            dataset_name,
            version=dataset_version,
        )

        pickled_datasets_path = "./data/splits/"
        if not os.path.exists(pickled_datasets_path):
            os.mkdir(pickled_datasets_path)

        dataset_partitions = [
            ("X_train", X_train),
            ("X_benign_validate", X_benign_validate),
            ("X_anomaly_validate", X_anomaly_validate),
            ("X_benign_test", X_benign_test),
            ("X_anomaly_test", X_anomaly_test),
        ]
        for partition_name, data in dataset_partitions:
            full_name = dataset_name.get_full_name(dataset_version)
            with open(
                    f"{pickled_datasets_path}{partition_name}_{full_name}.pkl",
                    "wb",
            ) as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Clear large variables from memory
        del (
            X_train,
            X_benign_validate,
            X_anomaly_validate,
            X_benign_test,
            X_anomaly_test,
        )
        gc.collect()
    print("Done")


if __name__ == "__main__":
    main()
