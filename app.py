import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# 1. Set up the UI Title and description
st.write("# 🌸 Iris Flower Predictor")
st.write("Adjust the sliders on the left to measure your flower, and the model will predict its species!")

# 2. Add an image dictionary
# This maps the species name (from the dataset) to a URL of that flower
IMAGE_URLS = {
    "setosa": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
    "versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

# 3. Build the sidebar with sliders
st.sidebar.header('Flower Measurements')

def get_user_input():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 1.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
    features = pd.DataFrame({
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    })
    return features

user_data = get_user_input()

st.subheader('Your Measurements')
st.write(user_data)

# 4. Load data and train the model
iris = load_iris()
model = KNeighborsClassifier(n_neighbors=3)
model.fit(iris.data, iris.target)

# 5. Make a prediction based on the slider values
prediction = model.predict(user_data)
predicted_species = iris.target_names[prediction[0]]

# 6. Display the result and the corresponding image
st.subheader('Prediction')
st.write(f"Based on these measurements, the flower is an **Iris {predicted_species}**.")

# Display the image using the URL from our dictionary
st.image(IMAGE_URLS[predicted_species], caption=f"Iris {predicted_species.capitalize()}", use_container_width=True)


