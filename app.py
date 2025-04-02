import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Set page config
st.set_page_config(
    page_title="Social Network Ads Purchase Prediction",
    page_icon="📊",
    layout="wide"
)

# Product requirements
PRODUCT_REQUIREMENTS = {
    'Smartphone': {'min_salary': 60000, 'min_age': 18},
    'Laptop': {'min_salary': 100000, 'min_age': 18},
    'Tablet': {'min_salary': 80000, 'min_age': 18},
    'Smartwatch': {'min_salary': 50000, 'min_age': 18},
    'Car': {'min_salary': 1000000, 'min_age': 18},
    'House': {'min_salary': 2500000, 'min_age': 18}
}

# Load the trained model, scaler, and label encoder
@st.cache_resource
def load_model():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    return model, scaler, label_encoder

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Social_Network_Ads.csv')
    # Add product categories (same as in model.py)
    products = list(PRODUCT_REQUIREMENTS.keys())
    data['Product'] = np.random.choice(products, size=len(data))
    return data

def prepare_data(data, label_encoder):
    """Prepare the data by encoding the Product column"""
    data['Product_Encoded'] = label_encoder.transform(data['Product'])
    return data

def check_eligibility(age, salary, product):
    """Check if the user is eligible to purchase the product"""
    requirements = PRODUCT_REQUIREMENTS[product]
    age_eligible = age >= requirements['min_age']
    salary_eligible = salary >= requirements['min_salary']
    
    if not age_eligible:
        return False, f"Age requirement not met. Minimum age required: {requirements['min_age']}"
    if not salary_eligible:
        return False, f"Salary requirement not met. Minimum salary required: ₹{requirements['min_salary']:,}"
    return True, "Eligible to purchase"

def main():
    st.title("📊 Social Network Ads Purchase Prediction")
    st.write("This app predicts whether a user will purchase a product based on their age, estimated salary, and product type.")

    # Load model and data
    model, scaler, label_encoder = load_model()
    data = load_data()
    
    # Prepare data by encoding the Product column
    data = prepare_data(data, label_encoder)

    # Create three columns for input
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
    
    with col2:
        salary = st.number_input("Estimated Salary (₹)", min_value=0, max_value=5000000, value=50000)
    
    with col3:
        product = st.selectbox(
            "Product",
            options=list(PRODUCT_REQUIREMENTS.keys())
        )

    # Display product requirements
    st.subheader("Product Requirements")
    requirements = PRODUCT_REQUIREMENTS[product]
    st.write(f"Minimum Age: {requirements['min_age']}")
    st.write(f"Minimum Salary: ₹{requirements['min_salary']:,}")

    # Make prediction
    if st.button("Check Eligibility"):
        # Check eligibility first
        eligible, message = check_eligibility(age, salary, product)
        
        if eligible:
            # Prepare input data
            product_encoded = label_encoder.transform([product])[0]
            input_data = np.array([[age, salary, product_encoded]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            purchase_probability = probabilities[1]
            
            # Display results
            st.subheader("Prediction Results")
            if prediction == 1:
                st.success(f"This user is eligible and likely to purchase the {product}! 🎉")
            else:
                st.warning(f"This user is eligible but unlikely to purchase the {product}.")
            
            st.write(f"Purchase Probability: {purchase_probability:.2%}")
            st.progress(float(purchase_probability))
        else:
            st.error(message)

    # Model Performance Section
    st.subheader("Model Performance")
    
    # Calculate and display metrics
    X = data[['Age', 'EstimatedSalary', 'Product_Encoded']]
    y = data['Purchased']
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.2%}")
    
    with col2:
        st.metric("Precision (Purchase)", f"{report['1']['precision']:.2%}")
    
    with col3:
        st.metric("Recall (Purchase)", f"{report['1']['recall']:.2%}")

    # Visualization section
    st.subheader("Data Visualization")
    
    # Create scatter plot with product information
    fig = px.scatter(data, x='Age', y='EstimatedSalary', 
                    color='Purchased',
                    symbol='Product',
                    title='Purchase Distribution by Age, Salary, and Product',
                    labels={'Purchased': 'Purchase Status'},
                    color_discrete_map={0: 'red', 1: 'green'})
    
    # Add the current prediction point if available
    if 'input_scaled' in locals():
        input_original = scaler.inverse_transform(input_scaled)
        fig.add_trace(
            go.Scatter(
                x=[input_original[0][0]],
                y=[input_original[0][1]],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='star',
                    color='yellow',
                    line=dict(color='black', width=2)
                ),
                name='Current Prediction'
            )
        )
    
    st.plotly_chart(fig, use_container_width=True)

    # Product-wise purchase rates
    st.subheader("Product-wise Purchase Rates")
    product_stats = data.groupby('Product')['Purchased'].agg(['mean', 'count']).reset_index()
    product_stats.columns = ['Product', 'Purchase Rate', 'Total Samples']
    fig_product = px.bar(product_stats, x='Product', y='Purchase Rate',
                        title='Purchase Rate by Product',
                        text='Total Samples',
                        labels={'Purchase Rate': 'Purchase Rate'})
    fig_product.update_traces(texttemplate='%{text} samples', textposition='outside')
    st.plotly_chart(fig_product, use_container_width=True)

    # Add model information
    with st.expander("Model Information"):
        st.write("""
        This model uses a Support Vector Machine (SVM) with RBF kernel to predict whether a user will purchase a product.
        
        Features used:
        - Age
        - Estimated Salary
        - Product Type
        
        Product Requirements:
        - Smartphone: Age ≥ 18, Salary ≥ ₹60,000
        - Laptop: Age ≥ 18, Salary ≥ ₹1,00,000
        - Tablet: Age ≥ 18, Salary ≥ ₹80,000
        - Smartwatch: Age ≥ 18, Salary ≥ ₹50,000
        - Car: Age ≥ 18, Salary ≥ ₹10,00,000
        - House: Age ≥ 18, Salary ≥ ₹25,00,000
        
        The model has been trained on a dataset of social network users and their purchase behavior.
        
        Performance Metrics:
        - Cross-validation is used to ensure model generalization
        - Regular monitoring of training vs. test accuracy to prevent overfitting
        - Product-specific analysis to understand purchase patterns
        """)

if __name__ == "__main__":
    main() 