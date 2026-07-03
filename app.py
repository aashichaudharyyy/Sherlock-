import streamlit as st
import pandas as pd

def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

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
    st.info("Please Upload a CSV file to let sherlock hop in 🚀")
else:
    df = pd.read_csv(file)
    #st.dataframe(df)
    rows, cols = df.shape
    missing_vals = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    #st.write(f"Sherlock sees **{rows}** rows and **{cols}** columns.")
    #st.write(f"Missing values: **{missing_vals}**")
    #st.write(f"Duplicate rows: **{duplicate_rows}**")

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
            # st.dataframe(missing)
            for col in missing.index:
                with st.expander(f"{col} has {missing[col]} missing values"):
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        st.write("Column Type : Date\n\nImputation method suggested: Forward fill (ffill) or Backward fill (bfill)")
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        st.write("Column Type : Numerical")
                        if abs(df[col].skew()) < 0.5:
                            st.write(f"Skewness: {df[col].skew():.2f}\n\nDistribution Type: Normal\n\nImputation method suggested: Mean Imputation")
                        else:
                            st.write(f"Skewness: {df[col].skew():.2f}\n\nDistribution Type: Skewed\n\nImputation method suggested: Median Imputation")
                    elif df[col].dtype == 'object':
                        st.write("Column Type : Categorical\n\nImputation method suggested: Mode Imputation")
        
        st.divider()

        st.subheader("Duplicate Rows")

        duplicates = df.duplicated().sum()
        duplicate_index = df[df.duplicated(keep=False)].index.tolist()

        if duplicates == 0:
            st.success("No duplicate rows found.")
        else:
            st.warning(f"{duplicates} duplicate rows found.")
            with st.expander("🔍 View Duplicate Records"):
                for idx in duplicate_index:
                    with st.expander(f"Row {idx}"):
                        st.dataframe(df.loc[[idx]])
            st.info("Sherlock's Recommendation :\n- Remove duplicate rows if they were introduced accidentally.\n- Keep them if they represent legitimate repeated events such as transactions, logs or sensor readings.")

        st.divider()

        st.subheader("Constant Columns")

        constant_cols = df.columns[df.nunique() == 1]

        if len(constant_cols) == 0:
            st.success("No constant columns found.")
        else:
            st.warning(f"{len(constant_cols)} found : {list(constant_cols)}")
            st.info("Sherlock's Recommendation :\n- Remove these columns before training.\n- They have only one unique value (zero variance).\n- Such features do not help machine learning models distinguish between observations.")
        

    st.subheader("Column-wise summary")
    summary_df = pd.DataFrame({
    "Columns":df.columns,
    "Data Type": df.dtypes,
    "Missing Values": df.isnull().sum(),
    "Unique Values": df.nunique()
    })
    st.dataframe(summary_df,hide_index=True)

    st.divider()

    st.subheader("Outlier Detection")

    numerical_cols = df.select_dtypes(include="number").columns

    outlier_found = False

    def outlier(col):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

        return outliers, lower_bound, upper_bound

    # for col in numerical_cols:

    #     q1 = df[col].quantile(0.25)
    #     q3 = df[col].quantile(0.75)

    #     iqr = q3 - q1

    #     lower_bound = q1 - 1.5 * iqr
    #     upper_bound = q3 + 1.5 * iqr

    #     outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    for col in numerical_cols:

        outliers, lower_bound, upper_bound = outlier(col)

        if not outliers.empty:

            outlier_found = True

            with st.expander(f"{col} has {len(outliers)} outlier(s)"):

                st.write(f"**Lower Bound:** {lower_bound:.2f}")
                st.write(f"**Upper Bound:** {upper_bound:.2f}")

                st.write("**Outlier Values**")
                st.dataframe(outliers[[col]], hide_index=True)

                st.info(
                    "🧠 Sherlock's Recommendation:\n"
                    "- Investigate these values before removing them.\n"
                    "- If they are genuine observations, keep them.\n"
                    "- Otherwise consider capping or removing them using the IQR method."
                )

    if not outlier_found:
        st.success("No outliers detected in numerical columns.")

    target_col = st.selectbox("Select Target", df.columns)

    def distribution(target_col):
        class_count = df[target_col].value_counts()
        class_percent = df[target_col].value_counts(normalize=True) * 100

        return class_count, class_percent
    
    class_count, class_percent = distribution(target_col)

    st.bar_chart(class_count)

    st.write("### Class Distribution")
    st.dataframe(pd.DataFrame({
        "Count": class_count,
        "Percentage": class_percent.round(2)
    }))

    if class_percent.max() > 80:
        st.warning("Imbalanced Dataset")
        st.info("""
    Sherlock's Recommendation

    - Use Stratified Split
    - Use F1-score
    - Consider SMOTE or Class Weights
    """)
    else:
        st.success("Dataset is reasonably balanced")
        st.info("""
    Sherlock's Recommendation

    - Proceed with training
    - Use Stratified Split
    - Accuracy is acceptable (depending on the problem)
    """)
        
    st.subheader("Correlation Check")

    num_df = df.select_dtypes(include="number")
    corr = num_df.corr()

    st.dataframe(corr)

    high_corr = False
    results = []

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):

            value = corr.iloc[i, j]

            if abs(value) > 0.8:

                high_corr = True

                results.append({
                    "Feature 1": corr.columns[i],
                    "Feature 2": corr.columns[j],
                    "Correlation": round(value, 2)
                })

    if not high_corr:
        st.success("No highly correlated features found.")

    else:
        st.write("Highly Correlated Features")
        res = pd.DataFrame(results)
        st.dataframe(res, hide_index=True)

        st.info("""
    Sherlock's Recommendation

    - Consider removing one of the highly correlated features.
    - High correlation can lead to multicollinearity.
    - This is especially important for Linear and Logistic Regression models.
    """)
        
    st.subheader("Encoding Suggestions")

    cat_df = df.select_dtypes(include="object")

    for col in cat_df.columns:

        with st.expander(col):

            count_unique = df[col].nunique()

            if count_unique == 2:
                encoding = "Label Encoding"

            elif count_unique <= 10:
                encoding = "One-Hot Encoding"

            else:
                encoding = (
                    "High Cardinality detected.\n"
                    "Consider Target Encoding or Frequency Encoding.\n"
                    "Avoid One-Hot Encoding as it may create too many columns."
                )

            st.write(
                f"Unique Values: {count_unique}\n\n"
                f"Recommendation: {encoding}"
            )
    
    st.subheader("Scaling Suggestions")

    for col in numerical_cols:

        outliers = outlier(col)

        skewness = abs(df[col].skew())

        with st.expander(col):

            st.write(f"Skewness: {skewness:.2f}")

            if not outliers.empty:

                st.write("Outliers: Yes")
                st.write("Recommendation: RobustScaler")
                st.write(
                    "Reason: Outliers are present in this feature. "
                    "RobustScaler uses the median and IQR, making it less sensitive to extreme values."
                )

            elif skewness < 0.5:

                st.write("Outliers: No")
                st.write("Distribution: Approximately Normal")
                st.write("Recommendation: StandardScaler")
                st.write(
                    "Reason: The feature is approximately normally distributed "
                    "and does not contain significant outliers."
                )

            else:

                st.write("Outliers: No")
                st.write("Distribution: Skewed")
                st.write("Recommendation: MinMaxScaler")
                st.write(
                    "Reason: The feature is skewed but does not contain significant outliers."
                )

