# src/models/baselines.py
"""
baseline models pipeline.
Strictly excludes 'ndvi_spatial_lag' to eliminate target leakage and 
enforces a three-way chronological split (Train <= 2021, Val 2022-2023, Test > 2023)
to directly match the Deep Learning evaluation window.
Excludes OLS to remove structural redundancy.
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
    Lightweight wrapper for statsmodels GLM results.
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
        
    def prepare_data(self, df, train_split_year=None, val_end_year=None):
        """
        Performs a three-way chronological train/val/test split matching the DL dataset.
        Handles categorical soil features, imputes missing values strictly using training 
        data medians, and scales continuous features based strictly on training distributions.
        """
        if train_split_year is None:
            train_split_year = self.config.get('features', {}).get('train_split_year', 2021)
        if val_end_year is None:
            val_end_year = self.config.get('features', {}).get('val_end_year', 2023)
            
        print(f"Preparing matrices.")
        print(f"  Train split window : Years <= {train_split_year}")
        print(f"  Validation window  : Years {train_split_year + 1} to {val_end_year}")
        print(f"  Test split window  : Years > {val_end_year}")
        
        # 1. Start with a working copy so we don't mutate the raw input DataFrame
        df_working = df.copy()

        # 2. Dynamically construct temporal features if they aren't present
        if 'month_sin' not in df_working.columns or 'month_cos' not in df_working.columns:
            if 'month' in df_working.columns:
                print("Generating month_sin and month_cos from 'month' column...")
                df_working['month_sin'] = np.sin(2 * np.pi * df_working['month'] / 12.0)
                df_working['month_cos'] = np.cos(2 * np.pi * df_working['month'] / 12.0)
            else:
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

        # 3. One-hot encode soil classification categories
        df_encoded = pd.get_dummies(df_working, columns=['soil_snum'], drop_first=True, dtype=float)
        soil_cols = [col for col in df_encoded.columns if col.startswith('soil_snum_')]
        
        # 4. Define the feature set (excluding any spatial/target lags of NDVI)
        continuous_features = [
            'lst_driver_lag1', 'lst_driver_lag2', 'lst_driver_lag3',
            'log_precip_driver_lag1', 'log_precip_driver_lag2', 'log_precip_driver_lag3',
            'pop_density', 'twi',
            'month_sin', 'month_cos', 'year_trend'
        ]
        
        # key interactions features 
        interaction_features = []
        possible_interactions = ['lst_x_precip', 'twi_x_precip', 'twi_x_lst']
        for col in possible_interactions:
            if col in df_encoded.columns:
                interaction_features.append(col)
            elif 'lst_driver_lag1' in df_encoded.columns and 'log_precip_driver_lag1' in df_encoded.columns:
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
        
        # 5. Three-Way Chronological Train-Validation-Test Split
        train_mask = df_encoded['year'] <= train_split_year
        val_mask = (df_encoded['year'] > train_split_year) & (df_encoded['year'] <= val_end_year)
        test_mask = df_encoded['year'] > val_end_year
        
        X_train = df_encoded.loc[train_mask, self.features].copy()
        y_train = df_encoded.loc[train_mask, 'log_ndvi'].copy()
        
        X_val = df_encoded.loc[val_mask, self.features].copy()
        y_val = df_encoded.loc[val_mask, 'log_ndvi'].copy()
        
        X_test = df_encoded.loc[test_mask, self.features].copy()
        y_test = df_encoded.loc[test_mask, 'log_ndvi'].copy()
        
        # 6. Chronological Imputation strictly derived from X_train medians
        all_continuous = continuous_features + interaction_features
        train_medians = X_train[all_continuous].median()
        
        X_train[all_continuous] = X_train[all_continuous].fillna(train_medians)
        X_val[all_continuous] = X_val[all_continuous].fillna(train_medians)
        X_test[all_continuous] = X_test[all_continuous].fillna(train_medians)
        
        if soil_cols:
            X_train[soil_cols] = X_train[soil_cols].fillna(0.0)
            X_val[soil_cols] = X_val[soil_cols].fillna(0.0)
            X_test[soil_cols] = X_test[soil_cols].fillna(0.0)
            
        # 7. Standard Scaling fitted on the training set only
        X_train[all_continuous] = self.scaler.fit_transform(X_train[all_continuous])
        X_val[all_continuous] = self.scaler.transform(X_val[all_continuous])
        X_test[all_continuous] = self.scaler.transform(X_test[all_continuous])
        
        # 8. Final cast to float
        X_train = X_train.astype(float)
        X_val = X_val.astype(float)
        X_test = X_test.astype(float)
        y_train = y_train.astype(float)
        y_val = y_val.astype(float)
        y_test = y_test.astype(float)
        
        self.df_encoded = df_encoded
        
        # Return combined train+val as a unified matrix for models that don't need independent validation validation early stopping, 
        # but keep them clean for XGBoost early stopping.
        return X_train, X_val, X_test, y_train, y_val, y_test

    def run_baselines(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Trains and evaluates GLM Gaussian, Random Forest, and XGBoost baselines on the uniform test set.
        """
        results = {}
        
        # --- 1. GLM Gaussian ---
        print("Training GLM Gaussian...")
        # Add intercept specifically for statsmodels fit
        X_train_const = sm.add_constant(X_train)
        X_val_const = sm.add_constant(X_val, has_constant='add')
        X_test_const = sm.add_constant(X_test, has_constant='add')
        
        glm_model = sm.GLM(y_train, X_train_const, family=sm.families.Gaussian())
        self.glm_results = glm_model.fit()  
        
        wrapped_glm = StatsmodelsPredictionWrapper(self.glm_results)
        self.models['GLM_Gaussian'] = wrapped_glm
        
        print("\n" + "="*50)
        print("                 GLM GAUSSIAN SUMMARY")
        print("="*50)
        print(self.glm_results.summary())
        print("="*50 + "\n")
        
        # Evaluate GLM on both Val and Test
        glm_val_preds = self.glm_results.predict(X_val_const)
        glm_test_preds = self.glm_results.predict(X_test_const)
        
        results['GLM_Gaussian'] = {
            'Val_R2': r2_score(y_val, glm_val_preds),
            'Test_R2': r2_score(y_test, glm_test_preds),
            'Val_RMSE': np.sqrt(mean_squared_error(y_val, glm_val_preds)),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, glm_test_preds)),
            'MAE': mean_absolute_error(y_test, glm_test_preds)
        }

        # --- 2. Random Forest Regressor ---
        print("Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=400, 
            max_depth=20, 
            min_samples_leaf=75,
            max_features=0.3, 
            random_state=42, 
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        rf_val_preds = rf.predict(X_val)
        rf_test_preds = rf.predict(X_test)
        
        results['RandomForest'] = {
            'Val_R2': r2_score(y_val, rf_val_preds),
            'Test_R2': r2_score(y_test, rf_test_preds),
            'Val_RMSE': np.sqrt(mean_squared_error(y_val, rf_val_preds)),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, rf_test_preds)),
            'MAE': mean_absolute_error(y_test, rf_test_preds)
        }
        
        self.models['RandomForest'] = rf
        self.rf_feature_importance = pd.Series(
            rf.feature_importances_, 
            index=self.features
        ).sort_values(ascending=False)

        # --- 3. XGBoost Regressor ---
        print("Training XGBoost with Early Stopping on the Validation Set...")
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
        
        # Train strictly on X_train, evaluate early stopping rounds on true X_val
        xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        xgb_val_preds = xgb.predict(X_val)
        xgb_test_preds = xgb.predict(X_test)
        
        results['XGBoost'] = {
            'Val_R2': r2_score(y_val, xgb_val_preds),
            'Test_R2': r2_score(y_test, xgb_test_preds),
            'Val_RMSE': np.sqrt(mean_squared_error(y_val, xgb_val_preds)),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, xgb_test_preds)),
            'MAE': mean_absolute_error(y_test, xgb_test_preds)
        }
        
        self.models['XGBoost'] = xgb
        
        # Print Consolidated Performance Matrix 
        print("\n" + "="*85)
        print("         CORRECTED BASELINE PERFORMANCE ON TRUE CROSS-VALIDATED TIMESTEPS")
        print("="*85)
        print(f"{'Model':<15} {'Val R2':<10} {'Test R2':<10} {'R2 Delta':<10} {'Val RMSE':<10} {'Test RMSE':<10}")
        for model_name, metrics in results.items():
            r2_delta = metrics['Test_R2'] - metrics['Val_R2']
            print(f"{model_name:<15} {metrics['Val_R2']:.4f}     {metrics['Test_R2']:.4f}     {r2_delta:+.4f}    {metrics['Val_RMSE']:.4f}     {metrics['Test_RMSE']:.4f}")
        print("="*85)
        
        return results

    def evaluate_all(self, X_train, X_val, X_test, y_train, y_val, y_test):
        return self.run_baselines(X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    print("Baseline script successfully adapted to three-way split configuration.")