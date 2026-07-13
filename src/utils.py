import pandas as pd

def get_outlier_stats(df, col):
    """Return IQR-based outlier rows and bounds for a numeric column."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers, lower_bound, upper_bound


def build_recommendation_frame(mapping):
    """Convert a column-to-recommendation mapping into a displayable table."""
    if not mapping:
        return pd.DataFrame(columns=["Feature", "Recommendation"])
    return pd.DataFrame({
        "Feature": list(mapping.keys()),
        "Recommendation": list(mapping.values())
    })
