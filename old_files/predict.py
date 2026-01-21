import pandas as pd
from joblib import load

# Load the preprocessor and model from the saved file
preprocessor, model = load('trained_model.joblib')

# Define the dummy data with the exact same columns/features as 'X' in 'train.py'
dummy_data = {
    'Birim_Fiyat': [10.5, 20.3],
    'Satis_Adedi': [50, 75],
    'Bolge': ['Kuzey', 'Güney']
}

# Create a DataFrame from the dummy data
df = pd.DataFrame(dummy_data)

# Cast categorical features to category dtype if necessary
categorical_features = ['Bolge']
for col in categorical_features:
    df[col] = df[col].astype('category')

# Transform the dummy data using the preprocessor
X_transformed = preprocessor.transform(df)

# Make predictions using the model
predictions = model.predict(X_transformed)

# Print the product name and the predicted sales amount
product_names = ['Product1', 'Product2']
for i, prediction in enumerate(predictions):
    print(f'Product: {product_names[i]}, Predicted Sales Amount: {prediction:.2f}')