# Bengaluru House Price Prediction

This repository contains a web application for predicting house prices in Bengaluru, India, using a machine learning model. The application is built with Streamlit and provides users with an easy-to-use interface to input features such as total square feet, number of bathrooms, number of balconies, BHK, and location to predict the house price.

## Features

- **Predict House Price:** Input features like total square feet, number of bathrooms, number of balconies, BHK, and location to get a price prediction.
- **Data Visualization:** Visualize various aspects of the dataset, including price distribution, total square feet distribution, and more.
- **Interactive Plots:** Explore interactive plots to understand the relationship between features and house prices.

## Tech Stack

- **Python:** The core language used for the project.
- **Streamlit:** The framework used to build the web application.
- **Pandas:** For data manipulation and analysis.
- **Joblib:** To load the pre-trained machine learning model.
- **Matplotlib & Seaborn:** For creating static visualizations.
- **Plotly:** For creating interactive visualizations.

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/bengaluru-house-price-prediction.git
    cd bengaluru-house-price-prediction
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the necessary files:
    - `house_price_model.pkl`: Trained machine learning model.
    - `Cleaned_Bengaluru_House_Data.csv`: Cleaned dataset.

    Place these files in the project directory.

## Usage

To run the Streamlit application, use the following command:

```bash
streamlit run app.py
