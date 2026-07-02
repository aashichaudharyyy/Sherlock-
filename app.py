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
        duplicate_index = df[df.duplicated()].index.tolist()

        if duplicates == 0:
            st.success("No duplicate rows found.")
        else:
            st.warning(f"{duplicates} duplicate rows found.")
            st.dataframe(df.loc[duplicate_index])

        st.divider()

        st.subheader("Constant Columns")

        constant_cols = df.columns[df.nunique() == 1]

        if len(constant_cols) == 0:
            st.success("No constant columns found.")
        else:
            st.warning(list(constant_cols))
        

    st.subheader("Column-wise summary")
    summary_df = pd.DataFrame({
    "Columns":df.columns,
    "Data Type": df.dtypes,
    "Missing Values": df.isnull().sum(),
    "Unique Values": df.nunique()
    })
    st.dataframe(summary_df,hide_index=True)



