import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

# Suppress warnings for cleaner output (optional)
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('data/train.csv')

# Basic Exploratory Data Analysis
print("Dataset Shape:", data.shape)
print("Missing Values:\n", data.isnull().sum())

# Selecting Features and Target
features = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
    'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
    'parking', 'prefarea', 'furnishingstatus'
]
target = 'price'

X = data[features]
y = data[target]

# Define numerical and categorical features
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories']
categorical_features = [
    'mainroad', 'guestroom', 'basement', 'hotwaterheating', 
    'airconditioning', 'parking', 'prefarea', 'furnishingstatus'
]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predicting and Evaluating the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate R² and MAE
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("Training R² Score:", train_r2)
print("Test R² Score:", test_r2)
print("Test MAE:", test_mae)

# Save the model
joblib.dump(model, 'house_price_model.joblib')
print("Model saved successfully!")
