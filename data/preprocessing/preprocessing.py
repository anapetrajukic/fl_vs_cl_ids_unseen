import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


def print_anomaly_stats(df, operation_name):
    """Helper function to print anomaly statistics consistently"""
    if "Label" in df.columns:
        total_samples = len(df)
        anomaly_samples = (df["Label"] != 0).sum()
        benign_samples = (df["Label"] == 0).sum()
        anomaly_ratio = (
            (anomaly_samples / total_samples * 100) if total_samples > 0 else 0
        )
        print(
            f"After {operation_name}: {total_samples} total samples ({benign_samples} benign, {anomaly_samples} anomalous, {anomaly_ratio:.2f}% anomaly rate)"
        )
    else:
        print(f"After {operation_name}: {len(df)} samples (no Label column)")


def print_shape_and_samples(df, operation_name, num_samples=3):
    """Helper function to print shape and sample rows consistently"""
    print(f"After {operation_name}: Shape {df.shape}")
    if len(df) > 0:
        print(f"Sample rows:\n{df.head(num_samples)}")
        print()


def divide_into_benign_and_anomaly(X, y):
    # Print detailed label distribution
    print(f"Label distribution in y:")
    label_counts = y.value_counts().sort_index()
    print(f"  Full distribution: {dict(label_counts)}")

    # Convert to binary (0 = benign, 1 = anomaly)
    y_binary = (y != 0).astype(int)
    binary_counts = y_binary.value_counts().sort_index()
    print(f"  Binary distribution: {dict(binary_counts)}")

    X_benign = X[y == 0]
    X_anomaly = X[y != 0]
    print(
        f"Data split: {len(X_benign)} benign samples, {len(X_anomaly)} anomaly samples"
    )
    print(
        f"Anomaly ratio: {len(X_anomaly) / (len(X_benign) + len(X_anomaly)) * 100:.2f}%"
    )
    print(f"Benign data shape: {X_benign.shape}")
    print(f"Anomaly data shape: {X_anomaly.shape}")
    return X_anomaly, X_benign


def extract_label(df):
    # Assume the last column is the label
    label_column = "Label"
    X = df.drop(columns=[label_column])
    y = df[label_column]
    print(f"Extracted label column: '{label_column}'")
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    print(
        f"Feature columns: {list(X.columns[:5])}{'...' if len(X.columns) > 5 else ''}"
    )
    print(f"Label distribution:\n{y.value_counts()}")
    return X, y


def remove_network_specific_features(df):
    features_before = df.shape[1]

    # Define columns to remove explicitly
    columns_to_remove = [
        "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS",
        "IPV4_SRC_ADDR",
        "L4_SRC_PORT",
        "IPV4_DST_ADDR",
        "L4_DST_PORT",
    ]

    # Remove specified columns
    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    if existing_columns_to_remove:
        df = df.drop(columns=existing_columns_to_remove)
        print(f"Removed network-specific columns: {existing_columns_to_remove}")

    features_after = df.shape[1]
    print(
        f"Removed {features_before - features_after} network-specific features ({features_before} -> {features_after})"
    )
    print_shape_and_samples(df, "network-specific feature removal")
    return df


def remove_inf_values(df):
    """
    Remove rows containing infinite values from a DataFrame.

    This function identifies and removes any rows in the input DataFrame that
    contain positive or negative infinite values in any column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame that may contain infinite values.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with rows containing infinite values removed.
        The original DataFrame structure and column names are preserved.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1, np.inf, 3], 'B': [4, 5, -np.inf]})
    >>> clean_df = remove_inf_values(df)
    >>> print(clean_df)
       A  B
    0  1  4

    Notes
    -----
    This function does not modify the original DataFrame in-place.
    It returns a copy with the infinite value rows filtered out.
    """
    # Count infinite values before removal
    inf_mask = df.isin([np.inf, -np.inf])
    inf_count_per_column = inf_mask.sum()
    columns_with_inf = inf_count_per_column[inf_count_per_column > 0]
    inf_count = inf_mask.sum().sum()
    rows_with_inf = inf_mask.any(axis=1).sum()
    rows_before = len(df)

    # Remove rows containing inf values from dataset
    clean_mask = ~inf_mask.any(axis=1)
    df = df[clean_mask]
    rows_after = len(df)

    print(
        f"Infinite values removal: {inf_count} infinite values in {rows_with_inf} rows, removed {rows_before - rows_after} rows ({rows_before} -> {rows_after})"
    )
    if len(columns_with_inf) > 0:
        print(f"  Columns with infinite values: {dict(columns_with_inf)}")
    print_shape_and_samples(df, "infinite values removal")
    return df


def expand_tcp_flags(
        df,
        tcp_flags_columns=["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"],
):
    """
    Expand TCP flags from single 8-bit integers into 8 separate binary features each.

    TCP flags are represented as bits in the following order:
    Bit 0: FIN, Bit 1: SYN, Bit 2: RST, Bit 3: PSH,
    Bit 4: ACK, Bit 5: URG, Bit 6: ECE, Bit 7: CWR

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing TCP flags columns.
    tcp_flags_columns : list, default=['TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS']
        List of column names containing TCP flags as 8-bit integers.

    Returns
    -------
    pandas.DataFrame
        DataFrame with TCP flags expanded into 8 separate binary columns each
        and the original TCP flags columns removed.
    """
    # TCP flag names corresponding to each bit position
    flag_names = ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]

    # Define prefixes for each column type
    prefix_map = {
        "TCP_FLAGS": "TCP",
        "CLIENT_TCP_FLAGS": "CLIENT_TCP",
        "SERVER_TCP_FLAGS": "SERVER_TCP",
    }

    features_before = df.shape[1]
    columns_to_drop = []

    for tcp_flags_column in tcp_flags_columns:
        # Check if the column exists
        if tcp_flags_column not in df.columns:
            print(f"Warning: Column '{tcp_flags_column}' not found in DataFrame")
            continue

        # Get the appropriate prefix
        prefix = prefix_map.get(
            tcp_flags_column, tcp_flags_column.replace("_FLAGS", "")
        )

        # Extract each bit as a separate binary feature
        for i, flag_name in enumerate(flag_names):
            # Extract bit i (0-7) from the TCP flags value using apply for Series operations
            df[f"{prefix}_{flag_name}"] = df[tcp_flags_column].apply(
                lambda x: (int(x) >> i) & 1
            )

        columns_to_drop.append(tcp_flags_column)

    # Remove the original TCP flags columns
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    features_after = df.shape[1]
    print(
        f"TCP flags expansion: {features_before} -> {features_after} features ({features_after - features_before} added)"
    )
    print_shape_and_samples(df, "TCP flags expansion")
    return df


def remove_attack_column(dataframe, inplace=True):
    result = dataframe.drop(columns=["Attack"], inplace=inplace)
    print(f"Removed categorical 'Attack' column")

    if not inplace:
        print_shape_and_samples(result, "attack column removal")
        return result
    else:
        print_shape_and_samples(dataframe, "attack column removal")


def remove_nan_values(df):
    """
    Remove rows containing NaN values from a DataFrame with detailed logging.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with NaN rows removed
    """
    # Count NaN values before removal
    nan_count_per_column = df.isnull().sum()
    columns_with_nan = nan_count_per_column[nan_count_per_column > 0]
    nan_count = nan_count_per_column.sum()
    rows_with_nan = df.isnull().any(axis=1).sum()
    rows_before = len(df)

    # Remove NaN values from dataset
    df_cleaned = df.dropna()
    rows_after = len(df_cleaned)

    print(
        f"NaN removal: {nan_count} NaN values in {rows_with_nan} rows, removed {rows_before - rows_after} rows ({rows_before} -> {rows_after})"
    )
    if len(columns_with_nan) > 0:
        print(f"  Columns with NaN values: {dict(columns_with_nan)}")

    print_shape_and_samples(df_cleaned, "NaN removal")
    return df_cleaned


class ClippingStandardScaler(StandardScaler):
    def fit(self, X, y=None, **kwargs):
        X = np.clip(X, None, 10e9)
        return super().fit(X, y, **kwargs)

    def transform(self, X, **kwargs):
        X = np.clip(X, None, 10e9)
        return super().transform(X, **kwargs)


class ClippingRobustScaler(RobustScaler):
    def fit(self, X, y=None, **kwargs):
        X = np.clip(X, None, 10e9)
        return super().fit(X, y)

    def transform(self, X, **kwargs):
        X = np.clip(X, None, 10e9)
        return super().transform(X)


def analyze_duplicates(df, operation_name=""):
    """
    Analyze duplicate patterns in the DataFrame with detailed reporting.

    Args:
        df: Input DataFrame
        operation_name: Name of the operation for logging

    Returns:
        Dictionary with duplicate analysis results
    """
    total_rows = len(df)

    # Check for exact duplicates (all columns)
    duplicate_mask = df.duplicated(keep=False)
    num_duplicate_rows = duplicate_mask.sum()
    num_unique_duplicate_groups = (
        df[duplicate_mask].drop_duplicates().shape[0] if num_duplicate_rows > 0 else 0
    )

    # Check duplicates excluding Label column if it exists
    if "Label" in df.columns:
        feature_cols = [col for col in df.columns if col != "Label"]
        feature_duplicate_mask = df.duplicated(subset=feature_cols, keep=False)
        num_feature_duplicate_rows = feature_duplicate_mask.sum()

        # Check if feature duplicates have same or different labels
        if num_feature_duplicate_rows > 0:
            feature_dups = df[feature_duplicate_mask].copy()
            feature_dups_grouped = feature_dups.groupby(feature_cols)["Label"].nunique()
            same_label_groups = (feature_dups_grouped == 1).sum()
            diff_label_groups = (feature_dups_grouped > 1).sum()
        else:
            same_label_groups = diff_label_groups = 0
    else:
        feature_duplicate_mask = duplicate_mask
        num_feature_duplicate_rows = num_duplicate_rows
        same_label_groups = diff_label_groups = 0

    analysis = {
        "total_rows": total_rows,
        "exact_duplicate_rows": num_duplicate_rows,
        "exact_duplicate_groups": num_unique_duplicate_groups,
        "feature_duplicate_rows": num_feature_duplicate_rows,
        "same_label_duplicate_groups": same_label_groups,
        "different_label_duplicate_groups": diff_label_groups,
        "exact_duplicate_ratio": (
            (num_duplicate_rows / total_rows * 100) if total_rows > 0 else 0
        ),
        "feature_duplicate_ratio": (
            (num_feature_duplicate_rows / total_rows * 100) if total_rows > 0 else 0
        ),
    }

    print(f"Duplicate Analysis {operation_name}:")
    print(f"  Total rows: {analysis['total_rows']}")
    print(
        f"  Exact duplicates: {analysis['exact_duplicate_rows']} rows ({analysis['exact_duplicate_ratio']:.2f}%) in {analysis['exact_duplicate_groups']} groups"
    )
    print(
        f"  Feature duplicates: {analysis['feature_duplicate_rows']} rows ({analysis['feature_duplicate_ratio']:.2f}%)"
    )
    if "Label" in df.columns:
        print(
            f"  Feature duplicates with same labels: {analysis['same_label_duplicate_groups']} groups"
        )
        print(
            f"  Feature duplicates with different labels: {analysis['different_label_duplicate_groups']} groups"
        )

    return analysis


def investigate_duplicate_samples(df, num_examples=3, operation_name=""):
    """
    Show examples of duplicate rows to understand patterns.

    Args:
        df: Input DataFrame
        num_examples: Number of duplicate groups to show
        operation_name: Name of the operation for logging
    """
    print(f"Duplicate Investigation {operation_name}:")

    if "Label" in df.columns:
        feature_cols = [col for col in df.columns if col != "Label"]

        # Group by features and show groups with multiple rows
        grouped = df.groupby(feature_cols)
        duplicate_groups = grouped.filter(lambda x: len(x) > 1).groupby(feature_cols)

        count = 0
        for name, group in duplicate_groups:
            if count >= num_examples:
                break
            print(f"\nDuplicate Group {count + 1} (Features: {name[:3]}...):")
            print(f"  Group size: {len(group)}")
            print(f"  Labels in group: {sorted(group['Label'].unique())}")
            print(f"  Sample rows:")
            print(group.head(min(5, len(group))).to_string(max_cols=10))
            count += 1
    else:
        # For data without labels, just show duplicate groups
        duplicate_mask = df.duplicated(keep=False)
        if duplicate_mask.any():
            duplicates = df[duplicate_mask].head(num_examples * 2)
            print(f"Sample duplicate rows:")
            print(duplicates.to_string(max_cols=10))


def remove_duplicate_rows(df):
    """
    Remove duplicate rows from the DataFrame with detailed analysis.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with duplicate rows removed
    """
    # Analyze duplicates before removal
    print("=" * 50)
    analyze_duplicates(df, "BEFORE removal")
    investigate_duplicate_samples(df, num_examples=2, operation_name="BEFORE removal")

    rows_before = len(df)
    df_cleaned = df.drop_duplicates()
    rows_after = len(df_cleaned)

    print(
        f"\nDuplicate removal: {rows_before - rows_after} duplicate rows removed ({rows_before} -> {rows_after})"
    )

    # Analyze remaining duplicates after removal (should be 0)
    if rows_after > 0:
        analyze_duplicates(df_cleaned, "AFTER removal")

    print("=" * 50)
    print_shape_and_samples(df_cleaned, "duplicate removal")
    return df_cleaned
