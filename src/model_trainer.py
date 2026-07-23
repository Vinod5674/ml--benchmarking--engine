import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self):
        self.models = {}

    def get_model(self, model_name, params):
        if model_name == "Logistic Regression":
            return LogisticRegression(C=params.get("C", 1.0), max_iter=1000)
        elif model_name == "Decision Tree":
            return DecisionTreeClassifier(max_depth=params.get("max_depth", 5), random_state=42)
        elif model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                random_state=42
            )
        else:
            raise ValueError("Unsupported Model Type")

    def train_and_evaluate(self, X, y, model_names, hyperparameters, test_size=0.2, random_state=42, cv_folds=5):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) < 10 else None
        )

        results = []
        fitted_models = {}

        for name in model_names:
            params = hyperparameters.get(name, {})
            model = self.get_model(name, params)

            # Fit Model
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            # Cross Validation Score
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1_weighted')

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
                "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0),
                "CV F1 Mean": np.mean(cv_scores),
                "CV F1 Std": np.std(cv_scores)
            })

            fitted_models[name] = model

        results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
        return results_df, fitted_models, X_train, X_test, y_train, y_test

    def get_feature_importance(self, model, feature_names):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return None

        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        return df_imp