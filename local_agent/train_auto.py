import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load the data
data = pd.read_csv('satis_verisi.csv')

# Preprocessing steps
numeric_features = ['Birim_Fiyat']
categorical_features = ['Urun']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Feature engineering for date-related features
data['Year'] = pd.to_datetime(data['Tarih']).dt.year
data['Month'] = pd.to_datetime(data['Tarih']).dt.month
data['Day'] = pd.to_datetime(data['Tarih']).dt.day
data['DayOfWeek'] = pd.to_datetime(data['Tarih']).dt.dayofweek

# Split the data into features and target variable
X = data.drop(columns=['Satis_Adedi', 'Tarih'])
y = data['Satis_Adedi']

# Impute missing values in Birim_Fiyat
X['Birim_Fiyat'] = X['Birim_Fiyat'].fillna(X['Birim_Fiyat'].mean())

# Split the data into training and validation sets using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_index, val_index in tscv.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the model
joblib.dump(pipeline, 'model.joblib')