# src/models/baselines.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

class BaselineModelEvaluator:
    def __init__(self, config=None):
        self.config = config
        self.models = {
            "OLS": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        self.glm_model = None
        self.features = []

    def prepare_data(self, df, train_split_year=2021):
        """Applies One-Hot Encoding to soil and splits data chronologically."""
        # One-hot encode the categorical soil variable
        df_encoded = pd.get_dummies(df, columns=['soil_snum'], drop_first=True)
        
        # Identify feature columns (everything except targets and time metadata)
        exclude_cols = {'year', 'month', 'log_ndvi'}
        self.features = [col for col in df_encoded.columns if col not in exclude_cols]
        
        # Chronological split to prevent temporal data leakage
        train_mask = df_encoded['year'] <= train_split_year
        test_mask = df_encoded['year'] > train_split_year
        
        X_train = df_encoded.loc[train_mask, self.features]
        y_train = df_encoded.loc[train_mask, 'log_ndvi']
        X_test = df_encoded.loc[test_mask, self.features]
        y_test = df_encoded.loc[test_mask, 'log_ndvi']
        
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Trains OLS, GLM, and Random Forest, returning metrics for comparison."""
        results = {}
        
        # 1. Train & Evaluate OLS
        self.models["OLS"].fit(X_train, y_train)
        ols_preds = self.models["OLS"].predict(X_test)
        results["OLS"] = self._compute_metrics(y_test, ols_preds)
        
        # 2. Train & Evaluate GLM (Gamma Family with Log Link)
        # Statsmodels is used for GLM to access robust statistical summary tables
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test, has_constant='add')
        
        # Ensure target is strictly positive for Gamma GLM
        y_train_shifted = y_train + abs(y_train.min()) + 0.01
        y_test_shifted = y_test + abs(y_train.min()) + 0.01
        
        self.glm_model = sm.GLM(y_train_shifted, X_train_const, family=sm.families.Gamma(link=sm.families.links.Log()))
        self.glm_results = self.glm_model.fit()
        glm_preds = self.glm_results.predict(X_test_const) - (abs(y_train.min()) + 0.01)
        results["GLM_Gamma"] = self._compute_metrics(y_test, glm_preds)
        
        # 3. Train & Evaluate Random Forest
        self.models["RandomForest"].fit(X_train, y_train)
        rf_preds = self.models["RandomForest"].predict(X_test)
        results["RandomForest"] = self._compute_metrics(y_test, rf_preds)
        
        return pd.DataFrame(results).T

    def _compute_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"R2_Score": r2, "RMSE": rmse}