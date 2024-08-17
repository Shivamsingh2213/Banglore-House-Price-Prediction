import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Load the trained model and feature names
model = joblib.load('house_price_model.pkl')
df = pd.read_csv('Cleaned_Bengaluru_House_Data.csv')
feature_names = df.drop(columns=['price', 'availability', 'area_type']).columns

# Extract location feature names
location_columns = [col for col in feature_names if col.startswith('location_')]
locations = [col.replace('location_', '') for col in location_columns]

# Function to predict house price
def predict_price(total_sqft, bath, balcony, bhk, location):
    loc_col = 'location_' + location
    data = {col: 0 for col in feature_names}
    data.update({'total_sqft': total_sqft, 'bath': bath, 'balcony': balcony, 'bhk': bhk})
    if loc_col in data:
        data[loc_col] = 1
    X = pd.DataFrame([data])
    # Ensure columns are in the same order as the model expects
    X = X[model.feature_names_in_]
    return model.predict(X)[0]

# Streamlit app
st.title("Bengaluru House Price Prediction")

# Input fields
total_sqft = st.number_input("Total Square Feet", min_value=500, max_value=10000, value=1000)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)
location = st.selectbox("Location", locations)

# Predict button
if st.button("Predict"):
    price = predict_price(total_sqft, bath, balcony, bhk, location)
    st.write(f"Predicted Price: ₹ {price:.2f} lakhs")

    # Plot the prediction
    fig = go.Figure(go.Indicator(
        mode = "number+gauge+delta",
        value = price,
        title = {'text': "Predicted House Price (in lakhs)"},
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    st.plotly_chart(fig)

# Visualize dataset
if st.checkbox("Show Data Visualization"):
    st.subheader("Dataset Visualizations")

    # Price distribution
    st.write("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['price'], kde=True, ax=ax)
    ax.set_xlabel('Price (in lakhs)')
    st.pyplot(fig)

    # Total square feet distribution
    st.write("Total Square Feet Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['total_sqft'], kde=True, ax=ax)
    ax.set_xlabel('Total Square Feet')
    st.pyplot(fig)

    # Number of bathrooms distribution
    st.write("Number of Bathrooms Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='bath', data=df, ax=ax)
    st.pyplot(fig)

    # Number of balconies distribution
    st.write("Number of Balconies Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='balcony', data=df, ax=ax)
    st.pyplot(fig)

    # Number of bedrooms distribution
    st.write("Number of Bedrooms (BHK) Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='bhk', data=df, ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if numeric_df.empty:
        st.write("No numeric data available for correlation analysis.")
    else:
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Price vs. Total Square Feet
    st.write("Price vs. Total Square Feet")
    fig = px.scatter(df, x='total_sqft', y='price', title='Price vs. Total Square Feet')
    st.plotly_chart(fig)

    # Price vs. Number of Bedrooms (BHK)
    st.write("Price vs. Number of Bedrooms (BHK)")
    fig = px.box(df, x='bhk', y='price', title='Price vs. Number of Bedrooms (BHK)')
    st.plotly_chart(fig)

    # Location Heatmap
    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.write("Location Heatmap")
        fig = px.scatter(df, x='longitude', y='latitude', color='price', title='Location Heatmap',
                         labels={'price': 'Price (in lakhs)'}, color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.example.com/background.jpg");
        background-size: cover;
        font-family: 'Arial';
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stNumberInput input, .stSelectbox select {
        border: 2px solid #4CAF50;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
