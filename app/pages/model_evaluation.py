import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae

import os, sys
sys.path.append(os.path.abspath("."))

from src.data_loader import load_data

st.title("📊 Model Evaluation - SVD")

# Load and prepare
ratings, movies, df = load_data()
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train
model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Evaluate
rmse_score = rmse(predictions, verbose=False)
mae_score = mae(predictions, verbose=False)

# Show metrics
st.metric("📉 RMSE", f"{rmse_score:.4f}")
st.metric("📐 MAE", f"{mae_score:.4f}")

# Error distribution
errors = [abs(p.r_ui - p.est) for p in predictions]
st.subheader("Error Distribution")
sns.histplot(errors, bins=30)
st.pyplot(plt)

# Actual rating distribution
plt.clf()
st.subheader("Actual Rating Distribution")
sns.histplot(df['rating'], bins=10, kde=True)
st.pyplot(plt)