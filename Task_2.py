import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Generate embeddings for the product descriptions
def generate_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a pre-trained model from Hugging Face
    embeddings = model.encode(data['Description'].tolist())
    return embeddings

# Function to recommend similar products
def recommend_similar_products(product_name, data, embeddings):
    # Check if product exists
    if product_name not in data['ProductName'].values:
        print(f"Product '{product_name}' not found in the dataset.")
        return []

    # Get the index of the product
    product_idx = data[data['ProductName'] == product_name].index[0]

    # Calculate similarities
    cosine_similarities = cosine_similarity(embeddings[product_idx].reshape(1, -1), embeddings).flatten()
    
    # Get indices of similar products
    similar_indices = cosine_similarities.argsort()[-6:-1][::-1]  # Get top 5 similar products

    # Return similar products
    return data['ProductName'].iloc[similar_indices].tolist()

# Main code
if __name__ == "__main__":
    file_path = 'NLP_Task_Dataset.csv'  # Change this to your dataset path

    # Load data
    data = load_data(file_path)

    # Generate embeddings
    embeddings = generate_embeddings(data)

    # Print available products
    print("Available Products:")
    print(data['ProductName'].unique())  # Print unique product names

    # User input
    product_name = input("Enter the product name to find similar products: ")
    
    # Recommend similar products
    similar_products = recommend_similar_products(product_name, data, embeddings)
    
    # Display similar products
    if similar_products:
        print(f"Similar products to '{product_name}':")
        for product in similar_products:
            print(f"- {product}")
