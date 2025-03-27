import streamlit as st
import pickle as pkl
import pandas as pd

st.set_page_config(layout="wide")
st.title("IPL Win Predictor")

# Load the pickled data and model
teams = pkl.load(open("team.pkl", "rb"))
cities = pkl.load(open("city.pkl", "rb"))
model = pkl.load(open("model.pkl", "rb"))

# First row and column for team and city selection
col1, col2, col3 = st.columns(3)
with col1:
    batting_team = st.selectbox("Select the batting team", sorted(teams))

with col2:
    bowling_team = st.selectbox("Select the bowling team", sorted(teams))

with col3:
    selected_city = st.selectbox("Select the host city", sorted(cities))

# Target score input
target = st.number_input("Target Score", min_value=0, max_value=720, step=1)

# Second row for score, overs, and wickets
col4, col5, col6 = st.columns(3)
with col4:
    score = st.number_input("Score", min_value=0, max_value=720, step=1)
with col5:
    overs = st.number_input("Overs Done", min_value=0, max_value=20, step=1)
with col6:
    wickets = st.number_input("Wickets Fell", min_value=0, max_value=10, step=1)

if st.button("Predict Probabilities"):
    # Calculate remaining runs, balls, and wickets
    runs_left = target - score
    ball_left = 120 - (overs * 6)  # Total balls in 20 overs = 120
    remaining_wickets = 10 - wickets  # Calculate remaining wickets
    crr = score / overs if overs > 0 else 0  # Calculate current run rate
    rrr = (runs_left * 6) / ball_left if ball_left > 0 else 0  # Calculate required run rate

    # Prepare input data for prediction
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'Score': [score],
        'Wickets': [remaining_wickets],  # Use the remaining wickets
        'Remaining Balls': [ball_left],
        'target_left': [runs_left],
        'crr': [crr],
        'rrr': [rrr],
    })

    # Predict probabilities using the model
    result = model.predict_proba(input_df)
    loss = result[0][0]  # Loss probability
    win = result[0][1]   # Win probability

    # Display the probabilities
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")




