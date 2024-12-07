import numpy as np
import pandas as pd

from src.visualizer.plot import save_day_distribution, save_time_distribution


class DataTransformer:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def transform(self) -> pd.DataFrame:
        self._melt_data()
        self._pivot_data()
        self._fix_types()
        self._fix_names()

        print(f"Data transformed: {self.data.shape}")
        return self.data
    
    def adjust_timestamps(self) -> pd.DataFrame:
        """
        Convert epoch timestamp in UTC to datetime objects in PST timezone.
        """
        self.data['created_at_timestamp'] = self.data["created_at"]
        self.data['created_at'] = pd.to_datetime(self.data['created_at'], unit='s', utc=True)
        self.data['created_at_pst'] = self.data['created_at'].dt.tz_convert('America/Los_Angeles')
        self.data['scraped_at_timestamp'] = self.data["scraped_at"]
        self.data['scraped_at'] = pd.to_datetime(self.data['scraped_at'], unit='s', utc=True)
        self.data['scraped_at_pst'] = self.data['scraped_at'].dt.tz_convert('America/Los_Angeles')

        print("Timestamps converted and timezone adjusted")
        return self.data
    
    def categorize_day_time(self) -> pd.DataFrame:
        """
        Categorize days of week in PST into 1 (Monday) to 7 (Sunday).

        Categorize time of day in PST into 4 categories: Night, Morning, Afternoon, Evening.

        Examines distributions.

        Merge Night and Evening into Evening category.
        """
        self._cat_days()
        self._cat_time()
        
        print("Days and times categorized")
        return self.data
    
    def aggregate_reels(self) -> pd.DataFrame:
        """
        Aggregate reels by each user-day-time bin.
        """
        # Aggregate by user-day-time combination
        self.data = self.data.groupby(
            by=['user_id', 'day_of_week_pst', 'categorized_created_at_pst'],
        ).agg(
            {
                'total_posts': 'mean',
                'followers': 'mean',
                'followings': 'mean',
                'likes_count': 'mean',
                'comments_count': 'mean',
                'video_view_count': 'mean',
                'video_duration': 'mean',
            }
        ).dropna().reset_index()

        # Replace 2-col bin with 1-col bin
        self.data['day_time_group'] = self.data['day_of_week_pst'].astype(str) + '_' + self.data['categorized_created_at_pst'].astype(str)

        print(f"Data aggregated: {self.data.shape}")
        return self.data
    
    def get_engagement_per_follower(self) -> pd.DataFrame:
        """
        Calculate engagement per follower.
        """
        # Devise engagement velocity metric
        self.data["EPF"] = (
            (self.data["likes_count"] + self.data["comments_count"]) / self.data["followers"]
        ) * 1000
        self.data["log_EPF"] = np.log1p(self.data["EPF"])

        print("Engagement per follower calculated")
        return self.data

    def _melt_data(self) -> None:
        """
        Melt the data to create a single column for each metric
        """
        self.data = self.data.melt(
            id_vars=[
                "user_id",
                "timestamp",
                "data.user_data.meta.total_posts_count",
                "data.user_data.meta.followers_count",
                "data.user_data.meta.followings_count",
            ],
            var_name="variable",
            value_name="value",
        )
        self.data[["reel_id", "field"]] = self.data["variable"].str.extract(r"reel\.(\d+)\.(.+)")
        self.data.sort_values(["user_id", "reel_id", "field"])

        print(f"Data melted into shape: {self.data.shape}")

    def _pivot_data(self) -> None:
        """
        Pivot data
        """
        self.data = self.data.pivot(
            index=[
                "user_id",
                "reel_id",
                "data.user_data.meta.total_posts_count",
                "data.user_data.meta.followers_count",
                "data.user_data.meta.followings_count",
                "timestamp",
            ],
            columns="field",
            values="value",
        ).reset_index()

    def _fix_types(self) -> None:
        # Reassign reel metrics types
        reel_metrics = {
            "has_audio": pd.BooleanDtype(),
            "video_view_count": pd.Int64Dtype(),
            "comments_count": pd.Int64Dtype(),
            "comments_disabled": pd.BooleanDtype(),
            "taken_at_timestamp": pd.Int64Dtype(),
            "likes_count": pd.Int64Dtype(),
            "video_duration": pd.Float64Dtype(),
        }

        for column, dtype in reel_metrics.items():
            self.data[column] = self.data[column].astype(dtype)
        
        self.data["reel_id"] = self.data["reel_id"].astype(pd.Int64Dtype())

    def _fix_names(self) -> None:
        """
        Rename index and columns
        """
        self.data.columns.name = None
        self.data = self.data.rename(
            columns={
                "timestamp": "scraped_at",
                "taken_at_timestamp": "created_at",
                "data.user_data.meta.total_posts_count": "total_posts",
                "data.user_data.meta.followers_count": "followers",
                "data.user_data.meta.followings_count": "followings",
            }
        )

    def _cat_days(self) -> None:
        """
        Categorize days of week in PST into 1 (Monday) to 7 (Sunday).
        """
        self.data['day_of_week_pst'] = self.data['created_at_pst'].dt.dayofweek + 1  # Adjust from 0-6 to 1-7
        self.data['day_of_week_pst'] = self.data['day_of_week_pst'].astype(pd.Int64Dtype())

        # Plot distribution of `day_of_week_pst`
        save_day_distribution(self.data)
        
        print(f"Day of week categorized: {self.data.shape}")

    def _cat_time(self) -> None:
        """
        Categorize time of day in PST into 4 categories: Night, Morning, Afternoon, Evening.

        Examines distributions.

        Merge Night and Evening into Evening category.
        """
        # Cut in 4 bins
        self.data['categorized_created_at_pst'] = pd.cut(
            self.data['created_at_pst'].dt.hour,
            bins=[-1, 4, 11, 16, 23],
            labels=['Night', 'Morning', 'Afternoon', 'Evening']
        )

        # Plot distribution of `time_category_pst`
        # Plot the initial cut
        save_time_distribution(self.data, "initial")

        # Merge Night and Evening into Evening category due to imbalance
        self.data = self.data.replace('Night', 'Evening')
        save_time_distribution(self.data, "merged")

        print(f"Time of day categorized: {self.data.shape}")
    