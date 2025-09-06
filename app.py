import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, accuracy
from surprise.model_selection import train_test_split

# --------------------
# Load and Train Model
# --------------------
@st.cache_resource  # cache model so it doesn’t retrain every run
def train_model():
    data = Dataset.load_builtin('ml-100k')
    trainset, testset = train_test_split(data, test_size=0.2)

    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    return algo, rmse, mae

algo, rmse, mae = train_model()

# --------------------
# Streamlit UI
# --------------------
st.title("Movie Recommendation System")
st.write("Built with **Surprise (SVD)** on MovieLens 100k dataset")

st.subheader("Model Performance")
st.write(f"**RMSE:** {rmse:.4f} | **MAE:** {mae:.4f}")

# User inputs
st.subheader("Predict a Rating")
user_id = st.text_input("Enter User ID (e.g., 196):", "196")
movie_id = st.text_input("Enter Movie ID (e.g., 302):", "302")

if st.button("Predict Rating"):
    prediction = algo.predict(user_id, movie_id)
    st.success(f"Predicted rating for User {user_id} on Movie {movie_id}: ⭐ {round(prediction.est, 2)}")

