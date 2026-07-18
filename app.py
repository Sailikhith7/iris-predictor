import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# 1. Set up the UI Title and description
st.write("# 🌸 Iris Flower Predictor")
st.write("Adjust the sliders on the left to measure your flower, and the model will predict its species!")

# 2. Build the sidebar with sliders
st.sidebar.header('Flower Measurements')

def get_user_input():
    # st.sidebar.slider(Label, Min, Max, Default)
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 1.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
    # Store the slider values in a dictionary, then convert to a DataFrame
    features = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })
    return features

# Capture the user's input
user_data = get_user_input()

# Display the user's current measurements on the main page
st.subheader('Your Measurements')
st.write(user_data)

# 3. Load data and train the model (happens instantly)
iris = load_iris()
model = KNeighborsClassifier(n_neighbors=3)
model.fit(iris.data, iris.target)

# 4. Make a prediction based on the slider values
prediction = model.predict(user_data)
predicted_species = iris.target_names[prediction[0]]

# 5. Display the result
st.subheader('Prediction')
st.write(f"Based on these measurements, the flower is an **Iris {predicted_species}**.")