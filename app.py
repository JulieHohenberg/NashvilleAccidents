import streamlit as st
import altair as alt
from vega_datasets import data
import kagglehub
import pandas as pd
import os

# Download the dataset
path = kagglehub.dataset_download("justinwilcher/nashville-accident-reports-jan-2018-apl-2025")

# Find the CSV file in the downloaded path
for file in os.listdir(path):
    if file.endswith('.csv'):
        csv_path = os.path.join(path, file)
        break

# Load into a pandas DataFrame
df = pd.read_csv(csv_path)

# Preview the DataFrame
print(df.head())