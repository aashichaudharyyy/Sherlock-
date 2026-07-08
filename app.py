from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


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

def handle_missing_values(df, recommendations):
    df = df.copy()
    for col, strategy in recommendations.get("missing", {}).items():
        if col in df.columns:
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode" and not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == "ffill":
                df[col] = df[col].ffill()
            elif strategy == "bfill":
                df[col] = df[col].bfill()
    return True, df


def handle_duplicate_rows(df, recommendations):
    df = df.copy()
    if recommendations.get("duplicates", False):
        df = df.drop_duplicates().reset_index(drop=True)
    return True, df


def remove_constant_columns(df, recommendations):
    df = df.copy()
    constants = recommendations.get("constants", [])
    df = df.drop(columns=[c for c in constants if c in df.columns], errors="ignore")
    return True, df


def handle_outliers(df, recommendations):
    df = df.copy()
    for col, bounds in recommendations.get("outliers", {}).items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            lower_bound = bounds.get("lower_bound")
            upper_bound = bounds.get("upper_bound")
            if lower_bound is not None and upper_bound is not None:
                df[col] = np.clip(df[col], lower_bound, upper_bound)
    return True, df


def apply_encoding(df, recommendations):
    df = df.copy()
    for col, strategy in recommendations.get("encoding", {}).items():
        if col in df.columns:
            if strategy == "Label Encoding":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            elif strategy == "High Cardinality":
                df = df.drop(columns=[col])
            elif strategy == "One-Hot Encoding":
                ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
                encoded_arr = ohe.fit_transform(df[[col]].astype(str))
                encoded_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=df.index)
                df = df.drop(columns=[col]).join(encoded_df)
    return True, df


def apply_scaling(df, recommendations):
    df = df.copy()
    for col, strategy in recommendations.get("scaling", {}).items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if strategy == "RobustScaler":
                scaler = RobustScaler()
            elif strategy == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
    return True, df

def recommend_models(problem_type, df, target_col):    
    models = []

    rows, cols = df.shape
    categorical = len(df.select_dtypes(include=["object", "category"]).columns)
    imbalance = df[target_col].value_counts(normalize=True).max()

    if problem_type == "Binary Classification":

        models.append({
            "Model": "Random Forest",
            "Rating": "★★★★★",
            "Reason": "Excellent baseline model. Handles mixed features and captures non-linear relationships."
        })

        models.append({
            "Model": "XGBoost",
            "Rating": "★★★★★",
            "Reason": "High accuracy and performs well on structured/tabular datasets."
        })

        models.append({
            "Model": "Logistic Regression",
            "Rating": "★★★★☆",
            "Reason": "Simple, fast and highly interpretable baseline classifier."
        })

        if rows < 10000:
            models.append({
                "Model": "SVM",
                "Rating": "★★★★☆",
                "Reason": "Suitable for small to medium-sized datasets."
            })

        models.append({
            "Model": "Decision Tree",
            "Rating": "★★★☆☆",
            "Reason": "Easy to interpret but prone to overfitting."
        })

    elif problem_type == "Multi-class Classification":

        models.append({
            "Model": "Random Forest",
            "Rating": "★★★★★",
            "Reason": "Strong performance on most multiclass tabular datasets."
        })

        models.append({
            "Model": "XGBoost",
            "Rating": "★★★★★",
            "Reason": "Excellent multiclass classification performance."
        })

        models.append({
            "Model": "LightGBM",
            "Rating": "★★★★☆",
            "Reason": "Very fast and efficient for larger datasets."
        })

        models.append({
            "Model": "KNN",
            "Rating": "★★★☆☆",
            "Reason": "Works well when the dataset is relatively small."
        })

        models.append({
            "Model": "Decision Tree",
            "Rating": "★★★☆☆",
            "Reason": "Simple and interpretable."
        })

    else:       # Regression

        models.append({
            "Model": "Random Forest Regressor",
            "Rating": "★★★★★",
            "Reason": "Captures complex non-linear relationships with minimal preprocessing."
        })

        models.append({
            "Model": "XGBoost Regressor",
            "Rating": "★★★★★",
            "Reason": "One of the strongest regression models for structured data."
        })

        models.append({
            "Model": "Linear Regression",
            "Rating": "★★★★☆",
            "Reason": "Fast baseline model for approximately linear relationships."
        })

        models.append({
            "Model": "Decision Tree Regressor",
            "Rating": "★★★☆☆",
            "Reason": "Easy to understand but can overfit."
        })

        if rows < 10000:
            models.append({
                "Model": "SVR",
                "Rating": "★★★☆☆",
                "Reason": "Suitable for smaller regression datasets."
            })

    return models

def select_model(selected_model, problem_type):

    classification_models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(random_state=42,eval_metric="logloss")
    }

    regression_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(random_state=42)
    }

    if "Classification" in problem_type:
        return classification_models[selected_model]

    return regression_models[selected_model]

def prepare_target(y, problem_type):

    if "Classification" in problem_type:
        if not pd.api.types.is_numeric_dtype(y):
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
            return y, encoder

    return y, None

st.set_page_config(
    page_title="Sherlock",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🕵️ Sherlock")
st.caption("Your AI Data Detective")
st.write(
    "Upload any CSV dataset and let Sherlock investigate its structure, "
    "quality, and readiness for Machine Learning."
)

st.divider()
st.subheader("📂 Upload Dataset")
file = st.file_uploader("Upload a CSV")

if not file:
    st.info("Please Upload a CSV file to let Sherlock hop in 🚀")
else:
    df = pd.read_csv(file)

    recommendations = {
        "missing": {},
        "duplicates": False,
        "constants": [],
        "outliers": {},
        "encoding": {},
        "scaling": {}
    }

    rows, cols = df.shape
    missing_vals = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    if st.button("Preview Dataset"):
        st.dataframe(df)

    st.subheader("Sherlock sees 🔎")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", rows)
    with col2:
        st.metric("Columns", cols)
    with col3:
        st.metric("Missing", missing_vals)
    with col4:
        st.metric("Duplicates", duplicate_rows)

    with st.expander("🔍 View Investigation Details"):
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            st.success("No missing values found.")
        else:
            for col in missing.index:
                with st.expander(f"{col} has {missing[col]} missing values"):
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        recommendations["missing"][col] = "ffill"
                        st.write(f"**Column Type**: Date")
                        st.write(f"**Missing Values**: {missing[col]}")
                        st.write("**Recommendation**: Forward fill for time-based continuity.")
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        skew = abs(df[col].skew())
                        if skew < 0.5:
                            recommendations["missing"][col] = "mean"
                            distribution = "Normal"
                            recommendation = "Mean"
                        else:
                            recommendations["missing"][col] = "median"
                            distribution = "Skewed"
                            recommendation = "Median"
                        st.write(f"**Column Type**: Numerical")
                        st.write(f"**Missing Values**: {missing[col]}")
                        st.write(f"**Distribution**: {distribution}")
                        st.write(f"**Recommendation**: Impute with the {recommendation}.")
                    else:
                        recommendations["missing"][col] = "mode"
                        st.write(f"**Column Type**: Categorical")
                        st.write(f"**Missing Values**: {missing[col]}")
                        st.write("**Recommendation**: Use the mode to preserve common values.")

        st.divider()
        st.subheader("Duplicate Rows")
        if duplicate_rows == 0:
            st.success("No duplicate rows found.")
        else:
            recommendations["duplicates"] = True
            st.warning(f"{duplicate_rows} duplicate rows found.")
            with st.expander("View Duplicate Records"):
                duplicate_df = df[df.duplicated(keep=False)].copy()
                if not duplicate_df.empty:
                    st.dataframe(duplicate_df, use_container_width=True)
                else:
                    st.info("No duplicate records to display.")
            st.write("**Sherlock's Recommendation**: Remove duplicate records to reduce bias and prevent overfitting.")

        st.divider()
        st.subheader("Constant Columns")
        constant_cols = df.columns[df.nunique() == 1].tolist()
        if len(constant_cols) == 0:
            st.success("No constant columns found.")
        else:
            recommendations["constants"] = constant_cols
            st.warning(f"{len(constant_cols)} constant columns found.")
            st.dataframe(pd.DataFrame({"Constant Column": constant_cols}), hide_index=True)

    st.subheader("Column-wise summary")
    summary_df = pd.DataFrame({
        "Columns": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(summary_df, hide_index=True)

    target_col = st.selectbox("Select Target", df.columns)

    st.divider()
    st.subheader("Outlier Detection")
    numerical_cols = df.select_dtypes(include="number").columns
    outlier_found = False

    for col in numerical_cols:
        outliers, lower_bound, upper_bound = get_outlier_stats(df, col)
        if not outliers.empty:
            outlier_found = True
            recommendations["outliers"][col] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
            with st.expander(f"{col} has {len(outliers)} outlier(s)"):
                st.write(f"**Lower Bound**: {lower_bound:.2f}")
                st.write(f"**Upper Bound**: {upper_bound:.2f}")
                st.write(f"**Number of Outliers**: {len(outliers)}")
                st.write("**Preview of Outlier Values**")
                st.dataframe(outliers[[col]].head(10), hide_index=True, use_container_width=True)
                st.write("**Sherlock's Recommendation**: Cap extreme values at the IQR bounds during preprocessing.")

    if not outlier_found:
        st.success("No outliers detected in numerical columns.")

    st.subheader("Encoding Suggestions")
    cat_df = df.select_dtypes(include=["object", "category"])
    high_cardinality_cols = []
    for col in cat_df.columns:
        if col == target_col:
            continue
        count_unique = df[col].nunique()
        if count_unique == 2:
            recommendations["encoding"][col] = "Label Encoding"
        elif count_unique <= 10:
            recommendations["encoding"][col] = "One-Hot Encoding"
        else:
            recommendations["encoding"][col] = "High Cardinality"

    encoding_frame = build_recommendation_frame(recommendations["encoding"])
    if not encoding_frame.empty:
        st.dataframe(encoding_frame, hide_index=True, use_container_width=True)
    else:
        st.info("No categorical encoding suggestions available.")


    st.subheader("Scaling Suggestions")
    for col in numerical_cols:
        if col == target_col:
            continue
        outliers, _, _ = get_outlier_stats(df, col)
        skewness = abs(df[col].skew())
        if not outliers.empty:
            recommendations["scaling"][col] = "RobustScaler"
        elif skewness < 0.5:
            recommendations["scaling"][col] = "StandardScaler"
        else:
            recommendations["scaling"][col] = "MinMaxScaler"

    scaling_frame = build_recommendation_frame(recommendations["scaling"])
    if not scaling_frame.empty:
        st.dataframe(scaling_frame, hide_index=True, use_container_width=True)
    else:
        st.info("No scaling suggestions available.")

    st.divider()
    if st.button("Prepare Dataset for ML"):
        total_steps = 6
        progress = st.progress(0)
        status_placeholder = st.empty()
        status_steps = [
            "Missing Values",
            "Duplicate Rows",
            "Constant Columns",
            "Outliers",
            "Encoding",
            "Scaling"
        ]
        status_map = {step: "⬜" for step in status_steps}
        status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
        status_placeholder.markdown(status_text)
        st.caption("Sherlock's processing engine is working...")

        df_processed = df.copy()

        success, df_processed = handle_missing_values(df_processed, recommendations)
        if success:
            progress.progress(int((1 / total_steps) * 100))
            status_map["Missing Values"] = "✓"
            status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
            status_placeholder.markdown(status_text)

        success, df_processed = handle_duplicate_rows(df_processed, recommendations)
        if success:
            progress.progress(int((2 / total_steps) * 100))
            status_map["Duplicate Rows"] = "✓"
            status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
            status_placeholder.markdown(status_text)

        success, df_processed = remove_constant_columns(df_processed, recommendations)
        if success:
            progress.progress(int((3 / total_steps) * 100))
            status_map["Constant Columns"] = "✓"
            status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
            status_placeholder.markdown(status_text)

        success, df_processed = handle_outliers(df_processed, recommendations)
        if success:
            progress.progress(int((4 / total_steps) * 100))
            status_map["Outliers"] = "✓"
            status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
            status_placeholder.markdown(status_text)

        success, df_processed = apply_encoding(df_processed, recommendations)
        if success:
            progress.progress(int((5 / total_steps) * 100))
            status_map["Encoding"] = "✓"
            status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
            status_placeholder.markdown(status_text)

        success, df_processed = apply_scaling(df_processed, recommendations)
        if success:
            progress.progress(int((6 / total_steps) * 100))
            status_map["Scaling"] = "✓"
            status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
            status_placeholder.markdown(status_text)

        st.success("Dataset successfully prepared for Machine Learning.")

        st.subheader("🕵️ Preprocessed Data Preview")
        st.dataframe(df_processed.head(10))

        csv_data = df_processed.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Prepared Dataset",
            data=csv_data,
            file_name="prepared_dataset.csv",
            mime="text/csv"
        )
        st.session_state["df_processed"] = df_processed

    target_dtype = df[target_col].dtype
    unique_values = df[target_col].nunique()
    if pd.api.types.is_numeric_dtype(df[target_col]):
        if unique_values <= 10:
            problem_type = "Classification"
        else:
            problem_type = "Regression"

    else:
        if unique_values == 2:
            problem_type = "Binary Classification"

        else:
            problem_type = "Multi-class Classification"

    st.subheader("Problem Type")
    st.success(problem_type)

    models = recommend_models(problem_type, df, target_col)
    st.subheader("Recommended Models")

    for model in models:
        with st.expander(f"{model['Rating']}  {model['Model']}"):
            st.write(model["Reason"])

    st.subheader("Select Model")

    selected_model = st.selectbox(
        "Choose a model to train",
        [model["Model"] for model in models]
    )

    for model in models:
        if model["Model"] == selected_model:
            st.info(
                f"⭐ {model['Rating']}\n\n"
                f"**Why Sherlock recommends it:**\n\n"
                f"{model['Reason']}"
            )

    if st.button("Train Model"):
        if "df_processed" not in st.session_state:
            st.error("Please click 'Prepare Dataset for ML' first.")
            st.stop()

        df_processed = st.session_state["df_processed"]

        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        y, target_encoder = prepare_target(y, problem_type)

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        st.success("Dataset splitted into training and testing dataset")
        
        model = select_model(selected_model, problem_type)
        model.fit(X_train,y_train)
        st.success(f"{selected_model} trainned & ready")
        
        predictions = model.predict(X_test)
        st.session_state["trained_model"] = model
        st.session_state["predictions"] = predictions
        st.session_state["y_test"] = y_test

