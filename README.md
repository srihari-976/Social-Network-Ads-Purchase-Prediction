# Social Network Ads Purchase Prediction App

This Streamlit application predicts whether a user will purchase a product based on their age and estimated salary using a Support Vector Machine (SVM) model.

## Features

- Interactive prediction interface
- Real-time purchase probability calculation
- Data visualization of the training dataset
- Model information and details
- Responsive design

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python model.py
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter the user's age (18-100)
2. Enter the user's estimated salary
3. Click "Predict Purchase" to get the prediction
4. View the visualization of the training data below

## Model Details

The model uses a Support Vector Machine (SVM) with RBF kernel and has been trained on the Social Network Ads dataset. Features used for prediction:
- Age
- Estimated Salary

## Deployment

This app can be deployed on Streamlit Cloud:
1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the app

## Requirements

- Python 3.7+
- See requirements.txt for package dependencies 