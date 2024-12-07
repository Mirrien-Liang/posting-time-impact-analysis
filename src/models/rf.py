import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from src.visualizer.plot import save_rf_residual_analysis

class RandomForest:
    def __init__(
        self,
        train_data: pd.DataFrame,
        x_cols: list[str],
        y_col: str,
    ) -> None:
        self.train_data = train_data
        self.x_cols = x_cols
        self.y_col = y_col
        self._X_train, self._X_valid, self._y_train, self._y_valid = train_test_split(
            self.train_data[x_cols],
            self.train_data[y_col],
            random_state=42,
        )

        # Model
        # Parameters obtained from Grid Search
        self.model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=42,
        )

    def train(self) -> None:
        print("Training Random Forest...")
        self.model.fit(self._X_train, self._y_train)

    def score(self) -> float:
        return self.model.score(self._X_valid, self._y_valid)

    def impute(self, missing_data: pd.DataFrame) -> np.ndarray:
        # Generate predictions on missing data
        imputed_log_views = self.model.predict(missing_data[self.x_cols])

        # Convert the logged views back
        imputed_values = np.expm1(imputed_log_views).round().astype(int)  # Since we used p1, we use m1 here

        return imputed_values
    
    def analyze_residuals(self) -> None:
        # Calculate residuals on train data
        predictions = self.model.predict(self._X_valid)
        residuals = self._y_valid - predictions

        save_rf_residual_analysis(residuals, predictions)

    def finetune(self) -> None:
        """
        Run with caution. Time consuming!
        """
        from sklearn.model_selection import GridSearchCV

        # Define hyperparameters for tuning
        param_grid_rf = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [1, 5],
            'min_samples_split': [2, 5],
        }

        # Initialize GridSearchCV
        grid_search_rf = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=param_grid_rf,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2,
        )
        grid_search_rf.fit(self._X_train, self._y_train)
        print(f'Grid Best Random Forest parameters: {grid_search_rf.best_params_}')
        print("="*33)
        # Grid Best Random Forest parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}

        # Try also RandomizedSearchCV
        from sklearn.model_selection import RandomizedSearchCV

        random_search_rf = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_distributions=param_grid_rf,
            n_iter=50,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2,
            random_state=42,
        )
        random_search_rf.fit(self._X_train, self._y_train)
        print(f'Randomized Best Random Forest parameters: {random_search_rf.best_params_}')
        # Randomized Best Random Forest parameters: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': None}

        # Assign the best model
        self.model = random_search_rf.best_estimator_
