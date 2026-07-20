# src/models/baselines.py
"""
Rebuilt baseline models pipeline.
Strictly excludes 'ndvi_spatial_lag' to eliminate target leakage and 
focuses on biophysical, topographic, and temporal drivers.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class StatsmodelsPredictionWrapper:
    """
    Lightweight wrapper for statsmodels GLM/OLS results.
    Automatically prepends the intercept constant during predict calls 
    to match the expected dimensions of the fitted model.
    """
    def __init__(self, fitted_model):
        self.fitted_model = fitted_model

    def predict(self, X):
        # Dynamically add constant if it is missing
        X_const = sm.add_constant(X, has_constant='add')
        return self.fitted_model.predict(X_const)

    def __getattr__(self, name):
        # Delegate any other attribute access (like summary) to the underlying model
        return getattr(self.fitted_model, name)


class NDVIBaselines:
    def __init__(self, config=None):
        """
        Initializes the baseline pipeline with a configuration dict.
        """
        self.config = config if config is not None else {}
        self.scaler = StandardScaler()
        self.features = []
        self.df_encoded = None
        self.glm_results = None  
        self.rf_feature_importance = None  
        self.models = {}  
        
    def prepare_data(self, df, train_split_year=None):
        """
        Performs chronological train/test splitting, handles categorical soil features,
        imputes missing values strictly using training data medians, and scales continuous features.
        
        Dynamically constructs temporal features and interaction terms if they are missing.
        This explicitly leaves out 'ndvi_spatial_lag' to prevent target leakage.
        """
        if train_split_year is None:
            # Fallback to 2021 if not specified in config
            train_split_year = self.config.get('features', {}).get('train_split_year', 2021)
            
        print(f"Preparing matrices. Train split year: <= {train_split_year}")
        
        # 1. Start with a working copy so we don't mutate the raw input DataFrame
        df_working = df.copy()

        # 2. Dynamically construct temporal features if they aren't present
        if 'month_sin' not in df_working.columns or 'month_cos' not in df_working.columns:
            if 'month' in df_working.columns:
                print("Generating month_sin and month_cos from 'month' column...")
                df_working['month_sin'] = np.sin(2 * np.pi * df_working['month'] / 12.0)
                df_working['month_cos'] = np.cos(2 * np.pi * df_working['month'] / 12.0)
            else:
                # If no month column is present at all, fall back to 0.0 values
                print("Warning: 'month' column not found! Initializing default 0.0 for sin/cos.")
                df_working['month_sin'] = 0.0
                df_working['month_cos'] = 0.0

        if 'year_trend' not in df_working.columns:
            if 'year' in df_working.columns:
                print("Generating year_trend relative to base year...")
                min_year = df_working['year'].min()
                df_working['year_trend'] = df_working['year'] - min_year
            else:
                print("Warning: 'year' column not found! Initializing year_trend to 0.0.")
                df_working['year_trend'] = 0.0

        # 3. One-hot encode soil classification categories safely (ensuring they are floats)
        df_encoded = pd.get_dummies(df_working, columns=['soil_snum'], drop_first=True, dtype=float)
        soil_cols = [col for col in df_encoded.columns if col.startswith('soil_snum_')]
        
        # 4. Define the feature set (excluding any spatial/target lags of NDVI)
        continuous_features = [
            'lst_driver_lag1', 'lst_driver_lag2', 'lst_driver_lag3',
            'log_precip_driver_lag1', 'log_precip_driver_lag2', 'log_precip_driver_lag3',
            'pop_density', 'twi',
            'month_sin', 'month_cos', 'year_trend'
        ]
        
        # Re-introduce key interactions from your July 7th run (if present in DataFrame)
        interaction_features = []
        possible_interactions = ['lst_x_precip', 'twi_x_precip', 'twi_x_lst']
        for col in possible_interactions:
            if col in df_encoded.columns:
                interaction_features.append(col)
            elif 'lst_driver_lag1' in df_encoded.columns and 'log_precip_driver_lag1' in df_encoded.columns:
                # Dynamically construct them if they aren't pre-computed
                if col == 'lst_x_precip':
                    df_encoded['lst_x_precip'] = df_encoded['lst_driver_lag1'] * df_encoded['log_precip_driver_lag1']
                    interaction_features.append('lst_x_precip')
                elif col == 'twi_x_precip':
                    df_encoded['twi_x_precip'] = df_encoded['twi'] * df_encoded['log_precip_driver_lag1']
                    interaction_features.append('twi_x_precip')
                elif col == 'twi_x_lst':
                    df_encoded['twi_x_lst'] = df_encoded['twi'] * df_encoded['lst_driver_lag1']
                    interaction_features.append('twi_x_lst')

        self.features = continuous_features + interaction_features + soil_cols
        
        # 5. Chronological Train-Test Split
        train_mask = df_encoded['year'] <= train_split_year
        test_mask = df_encoded['year'] > train_split_year
        
        X_train = df_encoded.loc[train_mask, self.features].copy()
        y_train = df_encoded.loc[train_mask, 'log_ndvi'].copy()
        X_test = df_encoded.loc[test_mask, self.features].copy()
        y_test = df_encoded.loc[test_mask, 'log_ndvi'].copy()
        
        # 6. Chronological Imputation (medians calculated strictly from X_train)
        all_continuous = continuous_features + interaction_features
        train_medians = X_train[all_continuous].median()
        X_train[all_continuous] = X_train[all_continuous].fillna(train_medians)
        X_test[all_continuous] = X_test[all_continuous].fillna(train_medians)
        
        if soil_cols:
            X_train[soil_cols] = X_train[soil_cols].fillna(0.0)
            X_test[soil_cols] = X_test[soil_cols].fillna(0.0)
            
        # 7. Standard Scaling (fitted strictly on training set)
        X_train[all_continuous] = self.scaler.fit_transform(X_train[all_continuous])
        X_test[all_continuous] = self.scaler.transform(X_test[all_continuous])
        
        # 8. Final cast to float
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        
        self.df_encoded = df_encoded
        
        return X_train, X_test, y_train, y_test

    def run_baselines(self, X_train, X_test, y_train, y_test):
        """
        Trains and evaluates OLS, GLM Gaussian, Random Forest, and XGBoost baselines.
        """
        results = {}
        
        # --- 1. OLS / GLM Gaussian ---
        print("Training OLS / GLM Gaussian...")
        # Add intercept specifically for statsmodels fit
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test, has_constant='add')
        
        glm_model = sm.GLM(y_train, X_train_const, family=sm.families.Gaussian())
        self.glm_results = glm_model.fit()  # Stored directly as raw statsmodels object
        
        # Save wrapped versions to the models dictionary to handle direct raw predict calls safely
        wrapped_glm = StatsmodelsPredictionWrapper(self.glm_results)
        self.models['GLM_Gaussian'] = wrapped_glm
        self.models['OLS'] = wrapped_glm
        
        print("\n" + "="*50)
        print("                 GLM GAUSSIAN SUMMARY")
        print("="*50)
        print(self.glm_results.summary())
        print("="*50 + "\n")
        
        # Predict GLM
        glm_preds = self.glm_results.predict(X_test_const)
        results['GLM_Gaussian'] = {
            'R2': r2_score(y_test, glm_preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, glm_preds)),
            'MAE': mean_absolute_error(y_test, glm_preds)
        }
        results['OLS'] = results['GLM_Gaussian']

        # --- 2. Random Forest Regressor ---
        print("Training Random Forest (this may take a few minutes)...")
        rf = RandomForestRegressor(
            n_estimators=400, 
            max_depth=20, 
            min_samples_leaf=75,
            max_features=0.3, # Constrained to prevent overfitting/OOM on big data
            random_state=42, 
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        results['RandomForest'] = {
            'R2': r2_score(y_test, rf_preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_preds)),
            'MAE': mean_absolute_error(y_test, rf_preds)
        }
        
        # Save to models dictionary
        self.models['RandomForest'] = rf
        
        # Map feature importances to feature names and sort them
        self.rf_feature_importance = pd.Series(
            rf.feature_importances_, 
            index=self.features
        ).sort_values(ascending=False)

        # --- 3. XGBoost Regressor ---
        print("Training XGBoost...")
        # Early stopping validation split from train set to protect the forward test set
        val_size = int(len(X_train) * 0.1)
        X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
        y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
        
        xgb = XGBRegressor(
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
        
        xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        xgb_preds = xgb.predict(X_test)
        results['XGBoost'] = {
            'R2': r2_score(y_test, xgb_preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, xgb_preds)),
            'MAE': mean_absolute_error(y_test, xgb_preds)
        }
        
        # Save to models dictionary
        self.models['XGBoost'] = xgb
        
        # --- Print Performance Matrix ---
        print("\n" + "="*50)
        print("         BASELINE MODEL PERFORMANCE ON FORWARD TEST SET")
        print("="*50)
        print(f"{'Model':<15} {'R2_Score':<10} {'RMSE':<10} {'MAE':<10}")
        for model_name, metrics in results.items():
            print(f"{model_name:<15} {metrics['R2']:.6f}   {metrics['RMSE']:.6f}   {metrics['MAE']:.6f}")
        print("="*50)
        
        return results

    # Alias evaluate_all to run_baselines to maintain compatibility with analyze_and_tune.py
    def evaluate_all(self, X_train, X_test, y_train, y_test):
        return self.run_baselines(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    # Test script loading and structures if executed directly
    print("Baseline script compiled successfully.")