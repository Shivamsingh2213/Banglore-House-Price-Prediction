import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

css = """
<style>
body {
    font-family: 'Arial', sans-serif;
    background-color: #f0f0f0;
}

h1 {
    color: #333;
    text-align: center;
}

.input-section, .visualizations-section {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.input-section {
    margin-top: 20px;
}

.title {
    color: #333;
    font-size: 2em;
    text-align: center;
}

.plot {
    margin: 20px 0;
}

img {
    width: 100%;
    height: auto;
    border-radius: 10px;
}

.sidebar .sidebar-content {
    font-size: 18px;
    color: #333;
}
</style>
"""


# Add custom CSS
# st.markdown(css, unsafe_allow_html=True)

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

# Display an image
# st.image('image.jpeg', caption='House Prices in Bengaluru', use_column_width=True)

# Input fields
st.markdown('<div class="input-section">', unsafe_allow_html=True)
total_sqft = st.number_input("Total Square Feet", min_value=500, max_value=10000, value=1000)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)
location = st.selectbox("Location", locations)
st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button("Predict"):
    price = predict_price(total_sqft, bath, balcony, bhk, location)
    st.write(f"Predicted Price: ₹ {price:.2f} lakhs")

    # Plot the prediction
    fig, ax = plt.subplots()
    sns.barplot(x=['Predicted Price'], y=[price], ax=ax)
    ax.set_ylabel('Price (in lakhs)')
    st.pyplot(fig)

# Visualize dataset
if st.checkbox("Show Data Visualization"):
    st.markdown('<div class="visualizations-section">', unsafe_allow_html=True)
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

    st.write("Correlation Heatmap")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Check if there are numeric columns
    if numeric_df.empty:
        st.write("No numeric data available for correlation analysis.")
    else:
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.write("Price vs. Total Square Feet")
    fig, ax = plt.subplots()
    sns.scatterplot(x='total_sqft', y='price', data=df, ax=ax)
    ax.set_xlabel('Total Square Feet')
    ax.set_ylabel('Price (in lakhs)')
    st.pyplot(fig)

    st.write("Price vs. Number of Bedrooms (BHK)")
    fig, ax = plt.subplots()
    sns.boxplot(x='bhk', y='price', data=df, ax=ax)
    ax.set_xlabel('Number of Bedrooms (BHK)')
    ax.set_ylabel('Price (in lakhs)')
    st.pyplot(fig)

    # Ensure you have latitude and longitude columns in your dataset
    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.write("Location Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x='longitude', y='latitude', hue='price', data=df, palette='viridis', ax=ax)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        st.pyplot(fig)

    st.write("Interactive Price vs. Total Square Feet")
    fig = px.scatter(df, x='total_sqft', y='price', title='Price vs. Total Square Feet')
    st.plotly_chart(fig)

    st.markdown('</div>', unsafe_allow_html=True)
