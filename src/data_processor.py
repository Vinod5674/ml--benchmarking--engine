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
        df_clean = df.copy()

        # FIX: Drop Identifiers & Names before encoding (RowNumber, CustomerId, Surname, etc.)
        cols_to_drop = []
        for col in df_clean.columns:
            if col != target_col:
                # Drop high cardinality string columns like Surname/Names
                if df_clean[col].dtype == 'object' and df_clean[col].nunique() > 10:
                    cols_to_drop.append(col)
                # Drop ID/Row columns
                elif 'id' in col.lower() or 'row' in col.lower() or 'number' in col.lower():
                    cols_to_drop.append(col)

        df_clean.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # 1. Target Column Separation
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]

        # Target Encoding
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            self.label_encoders['__target__'] = le

        # Identify Column Types
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # 2. Imputation
        if num_cols:
            self.num_imputer = SimpleImputer(strategy=num_strategy)
            X[num_cols] = self.num_imputer.fit_transform(X[num_cols])

        if cat_cols:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = self.cat_imputer.fit_transform(X[cat_cols])

        # 3. Categorical Encoding (Sirf low cardinality features like Gender, Geography par chalegi)
        if cat_cols:
            if categorical_action == 'onehot':
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            else:
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