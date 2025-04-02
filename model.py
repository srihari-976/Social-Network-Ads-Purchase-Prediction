import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def load_data():
    """Load the social network ads dataset"""
    # Download the dataset if it doesn't exist
    if not os.path.exists('Social_Network_Ads.csv'):
        import urllib.request
        url = "https://raw.githubusercontent.com/srihari-976/ML-models/main/Social%20Neworking%20Ads/Social_Network_Ads.csv"
        urllib.request.urlretrieve(url, "Social_Network_Ads.csv")
    
    # Load the dataset
    data = pd.read_csv('Social_Network_Ads.csv')
    
    # Add product categories with their minimum salary requirements
    products = ['Smartphone', 'Laptop', 'Tablet', 'Smartwatch', 'Car', 'House']
    min_salaries = {
        'Smartphone': 60000,
        'Laptop': 100000,
        'Tablet': 80000,
        'Smartwatch': 50000,
        'Car': 1000000,
        'House': 2500000
    }
    
    # Randomly assign products
    data['Product'] = np.random.choice(products, size=len(data))
    
    # Apply purchase rules
    data['Purchased'] = data.apply(
        lambda row: 1 if (
            row['Age'] >= 18 and 
            row['EstimatedSalary'] >= min_salaries[row['Product']]
        ) else 0,
        axis=1
    )
    
    return data

def prepare_data(data):
    """Prepare the data for training"""
    # Encode product categories
    le = LabelEncoder()
    data['Product_Encoded'] = le.fit_transform(data['Product'])
    
    # Select features
    X = data[['Age', 'EstimatedSalary', 'Product_Encoded']]
    y = data['Purchased']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def train_model(X_train_scaled, y_train):
    """Train the SVM model"""
    # Initialize and train the model with probability estimates enabled
    model = SVC(kernel='rbf', random_state=42, probability=True)
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):
    """Evaluate the model performance"""
    # Calculate training and test accuracy
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Calculate predictions and report
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred)
    
    return train_accuracy, test_accuracy, cv_scores, report

def save_model(model, scaler, label_encoder):
    """Save the trained model, scaler, and label encoder"""
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')

def main():
    # Load and prepare data
    data = load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, le = prepare_data(data)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    train_accuracy, test_accuracy, cv_scores, report = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Print results
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("\nClassification Report:")
    print(report)
    
    # Check for overfitting
    if train_accuracy > 0.95 and test_accuracy < 0.85:
        print("\nWARNING: Model might be overfitting!")
    elif train_accuracy < 0.7 and test_accuracy < 0.7:
        print("\nWARNING: Model might be underfitting!")
    
    # Save model and scaler
    save_model(model, scaler, le)
    print("\nModel, scaler, and label encoder saved successfully!")

if __name__ == "__main__":
    main() 