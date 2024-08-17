# Banglore-house-price-prediction
 Predicting the price of banlore houses


Bengaluru House Price Prediction App
This repository contains a Streamlit web application that predicts the price of houses in Bengaluru using a trained machine learning model. The app also provides several data visualizations to help users understand the underlying data better.

Features
House Price Prediction: Predict the price of a house based on inputs like total square feet, number of bathrooms, balconies, bedrooms (BHK), and location.
Data Visualization: Visualize the distribution of various features and their correlations, including:
Price distribution
Total square feet distribution
Number of bathrooms and balconies distribution
Number of bedrooms (BHK) distribution
Correlation heatmap
Price vs. Total Square Feet
Price vs. Number of Bedrooms (BHK)
Location heatmap
Custom Styling: The app is styled with custom CSS, including a background image, and styled input fields and buttons for a polished look.
Installation
To run this application locally, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/bengaluru-house-price-prediction.git
cd bengaluru-house-price-prediction
Install the required packages:
Make sure you have Python 3.x installed. Then, install the dependencies:

bash
Copy code
pip install streamlit pandas joblib matplotlib seaborn plotly
Download the dataset and model:

Ensure that Cleaned_Bengaluru_House_Data.csv is in the same directory as the Streamlit app.
Download or create a model file named house_price_model.pkl and place it in the same directory.
Run the app:

bash
Copy code
streamlit run app.py
Usage
Input: Enter the required details:

Total square feet
Number of bathrooms
Number of balconies
Number of bedrooms (BHK)
Select the location from the dropdown menu.
Prediction: Click the "Predict" button to see the predicted price of the house.

Visualization: Check the "Show Data Visualization" checkbox to explore various data visualizations.

File Structure
app.py: The main file containing the Streamlit application code.
house_price_model.pkl: The pre-trained model used for predictions.
Cleaned_Bengaluru_House_Data.csv: The dataset used for training the model and generating visualizations.
Contributing
Contributions are welcome! If you have any ideas, suggestions, or find any bugs, please feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
The dataset used in this project was sourced from Kaggle and was preprocessed for better model performance.
The project leverages multiple Python libraries such as Streamlit, Pandas, Matplotlib, Seaborn, and Plotly for the development and visualization aspects.

