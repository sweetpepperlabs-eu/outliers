from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

## I a file is not provided, the tool can be tested with a sample dataset.
def load_sample_data() -> pd.DataFrame:
    """Return a tiny sample dataset if the user does not upload one."""
    base_dir = Path(__file__).parent
    sample_path = base_dir / "data" / "time_observations.csv"
    try:
        return pd.read_csv(sample_path)
    except FileNotFoundError:
        st.error("Sample dataset not found. Please ensure the file exists.")
        return pd.DataFrame()


st.title("Outliers Detection")
st.write("This app is used to detect outliers in a dataset.")

st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data_source = "Uploaded file"
else:
    st.sidebar.info("No file uploaded. Using the built-in sample dataset.")
    df = load_sample_data()
    data_source = "Sample dataset"

# Show the data source and the first 5 rows of the dataset.
st.write(f"**Data source:** {data_source}")
st.dataframe(df.head())

st.subheader("Data Statistics")
st.write(df.describe(include="all"))

st.subheader("Axis Selection")
columns = df.columns.tolist()

if not columns:
    st.warning("The dataset does not contain any columns to select.")
else:
    default_x = columns[0]
    default_y = columns[1] if len(columns) > 1 else columns[0]

    x_axis = st.selectbox("Select X-axis column (usually time)", options=columns, index=columns.index(default_x))
    y_axis = st.selectbox("Select Y-axis column (usually observations)", options=columns, index=columns.index(default_y))

    st.write(f"Selected X-axis: `{x_axis}`")
    st.write(f"Selected Y-axis: `{y_axis}`")
    

st.subheader("Outliers Detection")
# Pick up quantile size
st.write("Pick up quantile size:")
quantile_value = st.slider("Quantile size", min_value=0.1, max_value=0.30, value=0.25, step=0.01)

st.write("Detecting outliers...")
lower_q, upper_q = quantile_value, 1 - quantile_value

lower_limit = df[y_axis].quantile(lower_q)
upper_limit = df[y_axis].quantile(upper_q)

st.write(f"Lower limit: {int(lower_limit)}")
st.write(f"Upper limit: {int(upper_limit)}")
outliers = df[(df[y_axis] < lower_limit) | (df[y_axis] > upper_limit)]
st.write(f"Number of outliers: {len(outliers)}")

# Plot the results
mask = (lower_limit <= df.observations) & (df.observations <= upper_limit)## create mask.
fig, ax = plt.subplots(figsize=(16, 6))## create a plot
fig.suptitle('Outlier Detection', size=20) ## title
sns.scatterplot(data=df, x='time', y='observations', hue=np.where(mask, 'No Outlier', 'Outlier'), ax=ax) ## apply color to outlier/no outlier
plt.tight_layout() ##makes sure that all variables and axes are readable

st.pyplot(fig)

data_clean = df[(df[y_axis].between(lower_limit, upper_limit))]
st.write(data_clean.head())
st.write(f"Number of data points: {len(data_clean)}")

csv_bytes = data_clean.to_csv(index=False).encode("utf-8")
downloaded = st.download_button(
    label="Download cleaned data (CSV)",
    data=csv_bytes,
    file_name="clean_data.csv",
    mime="text/csv",
    use_container_width=True,
)
if downloaded:
    st.balloons()
