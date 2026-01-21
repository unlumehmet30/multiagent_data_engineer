import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score
from joblib import dump

class ModelTrainer:
    def __init__(self):
        self.preprocessor = None
        self.model = None

    def train(self, data_file='satis_verisi.csv'):
        # Load the data
        df = pd.read_csv(data_file)

        # Drop rows where the target is NaN
        df = df.dropna(subset=['Satis_Adedi'])

        # Define categorical and numerical features
        categorical_features = ['Bolge']
        numerical_features = ['Satis_Adedi', 'Birim_Fiyat']

        # Create transformers for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

        # Define the model pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LGBMRegressor())
        ])

        # Split data into features and target
        X = df.drop(columns=['Tarih', 'Urun'])
        y = df['Satis_Adedi']

        # Define K-Fold Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Perform cross-validation
        rmse_scores = -cross_val_score(pipeline, X, y, cv=kf, scoring='neg_root_mean_squared_error')
        print(f'RMSE Scores: {rmse_scores}')
        print(f'Mean RMSE: {rmse_scores.mean()}')

        # Fit the model on the entire dataset
        pipeline.fit(X, y)

        self.preprocessor = preprocessor
        self.model = pipeline.named_steps['model']

    def save_model(self, model_file='trained_model.joblib'):
        dump((self.preprocessor, self.model), model_file)
        print(f'Model saved to {model_file}')

# Example usage
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
    trainer.save_model()