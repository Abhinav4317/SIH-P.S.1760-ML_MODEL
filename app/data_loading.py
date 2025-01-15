# app/data_loading.py
import os
import pickle
import json
import pandas as pd
from tensorflow.keras.models import load_model

# Load the models
model_1 = load_model("models/final_models/model_dnn_schemes.keras")
model_2 = load_model("models/final_models/triple_model_scheme.keras")

model_1_ins = load_model("models/final_models/model_dnn_insurence.keras")
model_2_ins = load_model("models/final_models/triple_model_insurence.keras")

# Load data
x_df = pd.read_csv("data/x.csv")
y_df = pd.read_csv("data/y_output.csv")
final_df = pd.read_csv("data/final_df.csv")

x_df_ins = pd.read_csv("data/x_insurence.csv")
y_df_ins = pd.read_csv("data/y_output_insurence.csv")

numeric_cols_1 = pickle.load(open("data/numeric_cols_scheme_dnn.pkl", "rb"))
numeric_cols_2 = pickle.load(open("data/triple_model_scheme_columns.pkl", "rb"))

numeric_cols_1_ins = pickle.load(open("data/numeric_cols_insurence_dnn.pkl", "rb"))
numeric_cols_2_ins = pickle.load(open("data/triple_model_insurence_columns.pkl", "rb"))

# Load past records and schemes info
past_sc_r = pd.read_csv("data/past_scheme_records.csv")
with open("data/post_office_schemes.json", "r") as f:
    post_office_schemes = json.load(f)

# Load datasets
demographics_df = pd.read_csv(
    "data/Updated_Dataset_with_Rural_and_Urban_Population.csv"
)  # Demographic data
agriculture_df = pd.read_csv("data/aggriculture_dataset.csv")  # Agricultural data

district_data = pd.read_csv("data/output.csv")

# Initialize Groq client
from groq import Groq

client = Groq(api_key="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")