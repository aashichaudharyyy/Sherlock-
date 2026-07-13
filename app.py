from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
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
import matplotlib.pyplot as plt
import joblib
import io
from src import preprocessing,recommendations,training,utils
import shap

st.set_page_config(
    page_title="Sherlock",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)


if "trained_models" not in st.session_state:
    st.session_state["trained_models"] = []

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

    prep_plan = {
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
                        prep_plan["missing"][col] = "ffill"
                        st.write(f"**Column Type**: Date")
                        st.write(f"**Missing Values**: {missing[col]}")
                        st.write("**Recommendation**: Forward fill for time-based continuity.")
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        skew = abs(df[col].skew())
                        if skew < 0.5:
                            prep_plan["missing"][col] = "mean"
                            distribution = "Normal"
                            recommendation = "Mean"
                        else:
                            prep_plan["missing"][col] = "median"
                            distribution = "Skewed"
                            recommendation = "Median"
                        st.write(f"**Column Type**: Numerical")
                        st.write(f"**Missing Values**: {missing[col]}")
                        st.write(f"**Distribution**: {distribution}")
                        st.write(f"**Recommendation**: Impute with the {recommendation}.")
                    else:
                        prep_plan["missing"][col] = "mode"
                        st.write(f"**Column Type**: Categorical")
                        st.write(f"**Missing Values**: {missing[col]}")
                        st.write("**Recommendation**: Use the mode to preserve common values.")

        st.divider()
        st.subheader("Duplicate Rows")
        if duplicate_rows == 0:
            st.success("No duplicate rows found.")
        else:
            prep_plan["duplicates"] = True
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
            prep_plan["constants"] = constant_cols
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
        outliers, lower_bound, upper_bound = utils.get_outlier_stats(df, col)
        if not outliers.empty:
            outlier_found = True
            prep_plan["outliers"][col] = {
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
            prep_plan["encoding"][col] = "Label Encoding"
        elif count_unique <= 10:
            prep_plan["encoding"][col] = "One-Hot Encoding"
        else:
            prep_plan["encoding"][col] = "High Cardinality"

    encoding_frame = utils.build_recommendation_frame(prep_plan["encoding"])
    if not encoding_frame.empty:
        st.dataframe(encoding_frame, hide_index=True, use_container_width=True)
    else:
        st.info("No categorical encoding suggestions available.")


    st.subheader("Scaling Suggestions")
    for col in numerical_cols:
        if col == target_col:
            continue
        outliers, _, _ = utils.get_outlier_stats(df, col)
        skewness = abs(df[col].skew())
        if not outliers.empty:
            prep_plan["scaling"][col] = "RobustScaler"
        elif skewness < 0.5:
            prep_plan["scaling"][col] = "StandardScaler"
        else:
            prep_plan["scaling"][col] = "MinMaxScaler"

    scaling_frame = utils.build_recommendation_frame(prep_plan["scaling"])
    if not scaling_frame.empty:
        st.dataframe(scaling_frame, hide_index=True, use_container_width=True)
    else:
        st.info("No scaling suggestions available.")

    st.divider()
    # ==========================================
    # MODIFICATION 1: Save the preprocessing plan to session state during training
    # Locate this block inside your original code where "Prepare Dataset for ML" finishes
    # ==========================================
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

        success, df_processed = preprocessing.handle_missing_values(df_processed, prep_plan)
        if success:
            progress.progress(int((1 / total_steps) * 100))
            status_map["Missing Values"] = "✓"
            status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
            status_placeholder.markdown(status_text)

        success, df_processed = preprocessing.handle_duplicate_rows(df_processed, prep_plan)
        if success:
            progress.progress(int((2 / total_steps) * 100))
            status_map["Duplicate Rows"] = "✓"
            status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
            status_placeholder.markdown(status_text)

        success, df_processed = preprocessing.remove_constant_columns(df_processed, prep_plan)
        if success:
            progress.progress(int((3 / total_steps) * 100))
            status_map["Constant Columns"] = "✓"
            status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
            status_placeholder.markdown(status_text)

        success, df_processed = preprocessing.handle_outliers(df_processed, prep_plan)
        if success:
            progress.progress(int((4 / total_steps) * 100))
            status_map["Outliers"] = "✓"
            status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
            status_placeholder.markdown(status_text)

        success, df_processed = preprocessing.apply_encoding(df_processed, prep_plan)
        if success:
            progress.progress(int((5 / total_steps) * 100))
            status_map["Encoding"] = "✓"
            status_text = "Sherlock's Processing Report\n\n" + "\n".join(f"{status_map[step]} {step}" for step in status_steps)
            status_placeholder.markdown(status_text)

        success, df_processed = preprocessing.apply_scaling(df_processed, prep_plan)
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
        
        # SAVE PIPELINE CONFIGURATIONS: Store recommendations dictionary into session state
        st.session_state["recommendations"] = prep_plan


    # ==========================================
    # REST OF TRAINING FLOW (Kept identical as per your code)
    # ==========================================
    target_dtype = df[target_col].dtype
    unique_values = df[target_col].nunique()
    if pd.api.types.is_numeric_dtype(df[target_col]):
        if unique_values == 2:
            problem_type = "Binary Classification"
        elif unique_values <= 10:
            problem_type = "Multi-class Classification"
        else:
            problem_type = "Regression"
    else:
        if unique_values == 2:
            problem_type = "Binary Classification"
        else:
            problem_type = "Multi-class Classification"

    st.subheader("Problem Type")
    st.success(problem_type)

    models = recommendations.recommend_models(problem_type, df, target_col)
    st.subheader("Recommended Models")

    for model in models:
        with st.expander(f"{model['Rating']}  {model['Model']}"):
            st.write(model["Reason"])

    st.subheader("Select Model")
    selected_model = st.selectbox("Choose a model to train", [model["Model"] for model in models])

    for model in models:
        if model["Model"] == selected_model:
            st.info(f"⭐ {model['Rating']}\n\n**Why Sherlock recommends it:**\n\n{model['Reason']}")

    if st.button("Train Model"):
        if "df_processed" not in st.session_state:
            st.error("Please click 'Prepare Dataset for ML' first.")
            st.stop()

        df_processed = st.session_state["df_processed"]

        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        y, target_encoder = training.prepare_target(y, problem_type)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.success("Dataset splitted into training and testing dataset")
        
        model = training.select_model(selected_model, problem_type)
        model.fit(X_train, y_train)
        st.success(f"{selected_model} trainned & ready")
        
        predictions = model.predict(X_test)
        st.session_state["trained_model"] = model
        st.session_state["predictions"] = predictions
        st.session_state["y_test"] = y_test
        st.session_state["processed_df"] = df_processed.copy()
        st.session_state["feature_columns"] = X.columns.tolist()
        st.session_state["target_column"] = target_col
        st.session_state["problem_type"] = problem_type
        try:
            st.session_state["target_encoder"] = target_encoder
        except NameError:
            st.session_state["target_encoder"] = None

        if hasattr(model, "feature_importances_"):
            st.subheader("Feature Importance")
            importance = model.feature_importances_
            importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})
            importance_df = importance_df.sort_values(by="Importance", ascending=False)
            st.dataframe(importance_df, hide_index=True, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance_df["Feature"], importance_df["Importance"])
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")
            st.pyplot(fig)

        y_test = st.session_state["y_test"]
        predictions = st.session_state["predictions"]

        if "Regression" in problem_type:
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            accuracy = r2
            precision = recall = f1 = float("nan")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{rmse:.4f}")
            c2.metric("MAE", f"{mae:.4f}")
            c3.metric("R2", f"{r2:.2%}")
            c4.metric("", "")
        else:
            try:
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
                recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
                f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
            except ValueError:
                try:
                    preds_round = np.rint(predictions).astype(y_test.dtype)
                    accuracy = accuracy_score(y_test, preds_round)
                    precision = precision_score(y_test, preds_round, average="weighted", zero_division=0)
                    recall = recall_score(y_test, preds_round, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, preds_round, average="weighted", zero_division=0)
                    predictions = preds_round
                    st.warning("Predictions were continuous; rounded to nearest class for metric calculation.")
                except Exception:
                    accuracy = precision = recall = f1 = float("nan")
                    st.warning("Unable to compute classification metrics for these predictions.")

            c1, c2, c3, c4 = st.columns(4)
            def fmt(x):
                return f"{x:.2%}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A"

            c1.metric("Accuracy", fmt(accuracy))
            c2.metric("Precision", fmt(precision))
            c3.metric("Recall", fmt(recall))
            c4.metric("F1 Score", fmt(f1))

        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        st.download_button(
            "Download Trained Model (.joblib)",
            data=model_buffer.getvalue(),
            file_name=f"{selected_model.replace(' ', '_').lower()}.joblib",
            mime="application/octet-stream"
        )

        st.session_state["trained_models"] = [
            m for m in st.session_state["trained_models"] if m.get("Model") != selected_model
        ]
        st.session_state["trained_models"].append({
            "Model": selected_model,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
    
    if st.session_state.get("trained_models"):
        leaderboard = pd.DataFrame(st.session_state["trained_models"])
        leaderboard = leaderboard.sort_values("Accuracy", ascending=False)
        st.dataframe(leaderboard, hide_index=True, use_container_width=True)

        best = leaderboard.iloc[0]
        st.markdown("🏆 Sherlock's Verdict")
        st.write(f"{best['Model']} is currently the best-performing model with {best['Accuracy']:.2%} accuracy.")

    # ==========================================
    # REFACTORED PREDICTION SECTION
    # ==========================================
    st.divider()
    st.header("🔮 Predict on New Dataset")

    prediction_file = st.file_uploader(
        "Upload CSV for Prediction",
        type=["csv"],
        key="prediction_csv"
    )

    if prediction_file is not None:
        if "trained_model" not in st.session_state:
            st.warning("Please train a model first.")
        elif "recommendations" not in st.session_state:
            st.warning("Preprocessing plan missing. Please regenerate training data splits.")
        else:
            # 1. Retrieve everything from session state
            model = st.session_state["trained_model"]
            feature_columns = st.session_state["feature_columns"]
            encoder = st.session_state.get("target_encoder")
            recommendations = st.session_state["recommendations"]

            # 2. Read evaluation batch
            raw_new_df = pd.read_csv(prediction_file)
            st.subheader("Uploaded Dataset")
            st.dataframe(raw_new_df.head())

            # 3. Clone for modifications during pipeline steps
            new_df = raw_new_df.copy()

            # If target column happens to be present in inference batch, drop it to match inference reality
            target_col_name = st.session_state.get("target_column")
            if target_col_name in new_df.columns:
                new_df = new_df.drop(columns=[target_col_name])

            # 4. Mirror preprocessing execution order (Omitting handle_duplicate_rows as instructed)
            _, new_df = preprocessing.handle_missing_values(new_df, recommendations)
            _, new_df = preprocessing.remove_constant_columns(new_df, recommendations)
            _, new_df = preprocessing.handle_outliers(new_df, recommendations)
            _, new_df = preprocessing.apply_encoding(new_df, recommendations)
            _, new_df = preprocessing.apply_scaling(new_df, recommendations)

            # 5. Schema alignment: align columns with training data and fill structure gaps with 0
            new_df = new_df.reindex(columns=feature_columns, fill_value=0)

            # 6. Execute model inference using preprocessed values
            predictions = model.predict(new_df)

            # 7. Convert targets back to string representations if classification categories exist
            if encoder is not None:
                predictions = encoder.inverse_transform(predictions)

            # 8. Render output
            result_df = raw_new_df.copy()
            result_df["Prediction"] = predictions

            st.subheader("Prediction Results")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
