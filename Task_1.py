import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(subset=['Description', 'SubCategory'], inplace=True)
    return data

# Step 2: Clean text
def clean_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    # Convert to lowercase
    text = text.lower()  # Convert to lowercase
    return text

# Step 3: Preprocess text
def preprocess_text(data):
    # Clean text using the clean_text function
    data['Description'] = data['Description'].apply(clean_text)  # Clean the descriptions
    data['ProductName'] = data['ProductName'].apply(clean_text)  # Clean product names
    # Combine ProductName and Description
    data['CombinedText'] = data['ProductName'] + ' ' + data['Description']
    return data

# Step 4: Encode labels (subcategories)
def encode_labels(data):
    label_encoder = LabelEncoder()
    data['SubCategoryEncoded'] = label_encoder.fit_transform(data['SubCategory'])
    return data, label_encoder

# Step 5: Build the model pipeline
def build_pipeline():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))),  # Use unigrams and bigrams
        ('classifier', RandomForestClassifier())  # Classifier
    ])
    return pipeline

# Step 6: Evaluate the model
def evaluate_model(pipeline, X_test, y_test, label_encoder):
    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    
    unique_labels = sorted(set(y_test))
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=label_encoder.classes_[unique_labels]))

    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Call the function to plot the confusion matrix
    plot_confusion_matrix(conf_matrix, label_encoder.classes_[unique_labels])

# Step 7: Plot confusion matrix as a heatmap
def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Step 8: Classify a new product description
def classify_product(pipeline, label_encoder, product_name, description):
    # Clean and preprocess both product name and description
    cleaned_product_name = clean_text(product_name)
    cleaned_description = clean_text(description)
    
    # Combine cleaned product name and description
    combined_text = f"{cleaned_product_name} {cleaned_description}"
    
    # Convert to DataFrame for the model
    input_data = pd.DataFrame([combined_text], columns=['CombinedText'])
    
    # Predict the subcategory
    predicted_encoded = pipeline.predict(input_data['CombinedText'].values)
    
    # Decode the label
    predicted_subcategory = label_encoder.inverse_transform(predicted_encoded)
    
    return predicted_subcategory[0]  # Return the predicted subcategory

# Step 9: Main process (extended to include real-time classification)
if __name__ == "__main__":
    file_path = 'NLP_Task_Dataset.csv'  # Use your provided dataset

    # Load and preprocess the data
    data = load_data(file_path)
    data = preprocess_text(data)
    data, label_encoder = encode_labels(data)

    # Get the features and labels
    X = data['CombinedText'].values  # Use the combined text for training
    y = data['SubCategoryEncoded'].values

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline
    pipeline = build_pipeline()

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    evaluate_model(pipeline, X_test, y_test, label_encoder)

    # Real-time classification example
    new_product_name = input("Enter the product name: ")
    new_product_description = input("Enter the product description to classify: ")
    predicted_subcategory = classify_product(pipeline, label_encoder, new_product_name, new_product_description)
    print(f"The predicted subcategory is: {predicted_subcategory}")

    # Analyze class distribution
    print("\nClass Distribution:")
    print(data['SubCategory'].value_counts())
