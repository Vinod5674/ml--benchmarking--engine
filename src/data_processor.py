import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self):
        self.num_imputer = None
        self.cat_imputer = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []

    def preprocess_data(self, df, target_col, num_strategy='median', scale_method='standard', categorical_action='onehot'):
        """
        Data Cleaning, Imputation, Encoding aur Scaling ka complete pipeline.
        """
        df_clean = df.copy()

        # 1. Target Column Separation
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]

        # Target Encoding (Agar target categorical/object type ka ho)
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            self.label_encoders['__target__'] = le

        # Identify Column Types
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # 2. Imputation (Missing Values Handle karna)
        if num_cols:
            self.num_imputer = SimpleImputer(strategy=num_strategy)
            X[num_cols] = self.num_imputer.fit_transform(X[num_cols])

        if cat_cols:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = self.cat_imputer.fit_transform(X[cat_cols])

        # 3. Categorical Encoding
        if cat_cols:
            if categorical_action == 'onehot':
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            else:  # Label Encoding
                for col in cat_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le

        # 4. Feature Scaling
        if scale_method == 'standard':
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)
        elif scale_method == 'minmax':
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns)

        self.feature_columns = X.columns.tolist()
        return X, y, self.feature_columns