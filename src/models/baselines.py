# src/models/baselines.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

class NDVIBaselines:
    def __init__(self, config=None):
        self.config = config
        self.models = {
            "OLS": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=400,
                max_depth=20,
                min_samples_leaf=75,
                max_features=0.3,
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=400,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                tree_method='hist',
                early_stopping_rounds=30
            )
        }
        self.glm_model = None
        self.glm_results = None
        self.scaler = StandardScaler()
        self.features = []
        self.rf_feature_importance = None

    def prepare_data(self, df, train_split_year=2021):
        """
        Prepares features with:
        - Cyclic month encoding (sin/cos)
        - Year as trend feature
        - Interaction terms (LST x precip, TWI x precip, TWI x LST)
        - One-hot encoding for soil
        - StandardScaler on continuous features
        - Chronological train/test split
        """
        # --- Cyclic month encoding ---
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # --- Year as trend feature ---
        df['year_trend'] = df['year']

        # --- Interaction terms ---
        df['lst_x_precip'] = df['lst_driver_lag1'] * df['log_precip_driver_lag1']
        df['twi_x_precip'] = df['twi'] * df['log_precip_driver_lag1']
        df['twi_x_lst']    = df['twi'] * df['lst_driver_lag1']

        # --- One-hot encode soil ---
        df_encoded = pd.get_dummies(df, columns=['soil_snum'], drop_first=True)

        # --- Feature columns ---
        exclude_cols = {'year', 'month', 'log_ndvi'}
        self.features = [col for col in df_encoded.columns if col not in exclude_cols]

        # --- Chronological split ---
        train_mask = df_encoded['year_trend'] <= train_split_year
        test_mask  = df_encoded['year_trend'] > train_split_year

        X_train = df_encoded.loc[train_mask, self.features].astype(np.float64)
        y_train = df_encoded.loc[train_mask, 'log_ndvi'].astype(np.float64)
        X_test  = df_encoded.loc[test_mask,  self.features].astype(np.float64)
        y_test  = df_encoded.loc[test_mask,  'log_ndvi'].astype(np.float64)

        # --- Standardize continuous features (fit on train only) ---
        continuous_cols = [
            'lst_driver_lag1', 'log_precip_driver_lag1', 'pop_density',
            'twi', 'year_trend', 'lst_x_precip', 'twi_x_precip', 'twi_x_lst'
        ]
        X_train[continuous_cols] = self.scaler.fit_transform(X_train[continuous_cols])
        X_test[continuous_cols]  = self.scaler.transform(X_test[continuous_cols])

        return X_train, X_test, y_train, y_test

    def evaluate_all(self, X_train, X_test, y_train, y_test):
        """Trains and evaluates OLS, GLM Gaussian, Random Forest, and XGBoost."""
        results = {}

        # 1. OLS
        print("Training OLS...")
        self.models["OLS"].fit(X_train, y_train)
        results["OLS"] = self._compute_metrics(y_test, self.models["OLS"].predict(X_test))

        # 2. GLM Gaussian (Identity link on log_ndvi — equivalent to OLS
        #    but provides full statsmodels summary: AIC, p-values, deviance)
        print("Training GLM Gaussian...")
        X_train_const = sm.add_constant(X_train)
        X_test_const  = sm.add_constant(X_test, has_constant='add')
        self.glm_model   = sm.GLM(
            y_train, X_train_const,
            family=sm.families.Gaussian(link=sm.families.links.Identity())
        )
        self.glm_results = self.glm_model.fit()
        results["GLM_Gaussian"] = self._compute_metrics(
            y_test, self.glm_results.predict(X_test_const)
        )

        # 3. Random Forest
        print("Training Random Forest (this may take a few minutes on HPC)...")
        self.models["RandomForest"].fit(X_train, y_train)
        results["RandomForest"] = self._compute_metrics(
            y_test, self.models["RandomForest"].predict(X_test)
        )

        # --- Extract and store RF feature importances ---
        self.rf_feature_importance = pd.Series(
            self.models["RandomForest"].feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        # 4. XGBoost with early stopping
        print("Training XGBoost (early stopping active, patience=30)...")
        self.models["XGBoost"].fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100
        )
        results["XGBoost"] = self._compute_metrics(
            y_test, self.models["XGBoost"].predict(X_test)
        )

        return results

    def _compute_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        return {"R2_Score": r2, "RMSE": rmse}