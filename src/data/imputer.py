import numpy as np
import pandas as pd

from src.models.rf import RandomForest
from src.visualizer.plot import save_missing_view_distribution, save_missing_view_over_time_elapsed, save_rf_corr_matrix_heatmap

class DataImputer:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def impute_views(self) -> pd.DataFrame:
        """
        Examine the missing view counts to see if random or systematic.
        
        Impute missing views with Random Forest.
        """
        print("Imputing missing views...")

        self._examine_randomness()
        self._preprocess()
        self._check_assumptions()
        self._impute_views()
        
        print("Missing views imputed")
        return self.data

    def _examine_randomness(self) -> None:
        """
        Check if the missingness was random or systematic.
        """
        # Check the missingness relative to key metrics
        # print(self.data.groupby(self.data['video_view_count'].isna())[["likes_count", "comments_count", "video_duration"]].describe())

        # Check if the missingness is related to the age of Reels
        save_missing_view_over_time_elapsed(self.data)

    def _preprocess(self) -> None:
        """
        Preprocess data for imputation
        """
        # Get time elasped in seconds
        self.data["time_elapsed_seconds"] = self.data["time_elapsed"].dt.total_seconds()

        # Encode has_audio
        self.data["has_audio"] = self.data["has_audio"].astype(pd.Int64Dtype())

        # One-Hot Encode 'day_of_week_pst' and 'categorized_created_at_pst'
        day_of_week_dummies = pd.get_dummies(
            self.data["day_of_week_pst"],
            prefix="day",
            drop_first=True,  # Use Monday as reference
        ).astype(pd.Int64Dtype())

        time_category_dummies = pd.get_dummies(
            self.data["categorized_created_at_pst"],
            prefix="time_cat",
            drop_first=True,  # Use Evening as reference
        ).astype(pd.Int64Dtype())

        # Concatenate the dummy variables to the dataframe
        self.data = pd.concat([self.data, day_of_week_dummies, time_category_dummies], axis=1)

        # Define features selected
        self.x_cols = [
            "total_posts",
            "followers",
            "followings",
            # "comments_count",  # Removed due to the multicollinearity issue with likes_count
            "has_audio",  # Encoded into binary 0 or 1
            "likes_count",  # .9 correlation with views; assuming no data leakage;
            "video_duration",
            "time_elapsed_seconds",
            # "day_of_week_pst",  # One-hot encoded
            # "categorized_created_at_pst",  # One-hot encoded
        ] + list(day_of_week_dummies.columns) + list(time_category_dummies.columns)
        
        self.y_col = "video_view_count"

        # Split missing and non-missing data
        self.train_data = self.data[self.data["video_view_count"].notna()].copy()
        self.missing_data = self.data[self.data["video_view_count"].isna()].copy()

        # Add logged view count
        self.y_col_transformed = 'video_view_count_log'
        self.train_data[self.y_col_transformed] = np.log1p(self.train_data[self.y_col])  # Get log(1+x) as view can be 0
    
    def _check_assumptions(self) -> None:
        """
        Check if the data meets the assumptions for Random Forest
        """
        save_missing_view_distribution(self.train_data, self.y_col)
        save_missing_view_distribution(self.train_data, self.y_col_transformed)
        save_rf_corr_matrix_heatmap(self.train_data, self.x_cols+[self.y_col_transformed])
        
    def _impute_views(self) -> None:
        rf = RandomForest(self.train_data, self.x_cols, self.y_col_transformed)
        # If to finetune hyperparameters, uncomment this
        # rf.finetune()

        # Fit model
        rf.train()

        # Get score
        print(f"Random Forest trained with R2 Score: {rf.score()}")

        # Predict missing views
        predictions = rf.impute(self.missing_data)

        # Update into dataframe
        self.data["is_view_count_imputed"] = self.data[self.y_col].isna().astype(pd.Int64Dtype())
        self.data.loc[self.data[self.y_col].isna(), self.y_col] = predictions

        # Do residual analysis
        rf.analyze_residuals()
