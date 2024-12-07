import numpy as np
import pandas as pd

from scipy import stats
from src.visualizer.plot import save_follower_distribution, save_time_elapsed_distribution

class DataFilter:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def filter_accounts(self) -> pd.DataFrame:
        self._filter_accounts()
        self._drop_columns()
        print(f"Accounts filtered: {self.data.shape}")
        return self.data
    
    def filter_reels(self) -> pd.DataFrame:
        self._filter_reels()
        print(f"Reels filtered: {self.data.shape}")
        return self.data

    def _filter_accounts(self) -> pd.DataFrame:
        """
        Filter rows:
        1. Failed jobs
        2. Account with no reels
        3. Private accounts
        4. Accounts with total posts < 12
        5. Accounts with followers < 50
        6. Influencer accounts
        """
        # Filter 1-5
        self.data = self.data.loc[
            (self.data['status'] == 'successful') &
            (self.data['data.user_data.meta.has_reel'] == True) &
            (self.data['data.user_data.meta.is_private'] == False) &
            (self.data['data.user_data.meta.total_posts_count'] >= 12) &
            (self.data['data.user_data.meta.followers_count'] > 50)
        ]
        print(f"Invalid accounts filtered: {self.data.shape}")

        # Identify influencers
        # print(self.data["data.user_data.meta.followers_count"].describe())

        # Plot distributions in followers
        save_follower_distribution(self.data)

        # Remove accounts with followers above the 98th percentile
        self.data = self.data[self.data['data.user_data.meta.followers_count'] <= self.data['data.user_data.meta.followers_count'].quantile(0.98)]

        # Evaluate effects
        # print(self.data["data.user_data.meta.followers_count"].describe())

        # Plot distributions in logged followers
        save_follower_distribution(self.data, suffix="98th")
        save_follower_distribution(self.data, log=True, suffix="98th")

        # Perform normal test on log followers, mild violation but good enough (p=0.018)
        print("Normality test on logged followers, p-value:", stats.normaltest(np.log1p(self.data["data.user_data.meta.followers_count"])).pvalue)

        print(f"Influncer accounts filtered: {self.data.shape}")

    def _drop_columns(self) -> pd.DataFrame:
        """
        Remove columns:
        1. Post-related columns
        2. Account type flags
        3. Other irrelevant columns
        """
        # Filter out columns with names containing "data.user_data.post"
        columns_to_drop = [col for col in self.data.columns if 'data.user_data.post' in col]

        # Filter out unary cols
        columns_to_drop.extend([
            'status',
            'data.user_data.meta.has_reel',
            'data.user_data.meta.is_private',
        ])

        # Filter out other irrelevant columns
        columns_to_drop.extend(
            [
                "data.user_data.meta.is_verified",
                "data.user_data.meta.is_business_account",
                "data.user_data.meta.is_professional_account",
            ]
        )

        # Drop columns
        self.data.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped irrelevant columns: {self.data.shape}")

    def _filter_reels(self) -> None:
        """
        Filter reels:
        1. Empty records with N/A posting time
        2. Comments disabled
        3. Video duration < 1 seconds or > 90 seconds (Instagram Reels typically have a max duration of 90 seconds)
        4. Reels with time elapsed within 7 days (detailed rationale in report)
        """
        # Drop empty records with N/A created_at timestamp
        self.data.dropna(subset=["created_at"], inplace=True)

        # Drop rows with comment disabled and remove this unary column
        self.data = self.data[self.data["comments_disabled"] == False].drop(columns=["comments_disabled"])

        # Drop rows with video duration < 1 or is NA or is > 90 seconds
        self.data = self.data[(self.data["video_duration"].between(1, 90, inclusive="both")) & (self.data["video_duration"]).notna()]

        # Add time elapsed col
        self.data["time_elapsed"] = self.data["scraped_at"] - self.data["created_at"]

        # Avoid changing original data for plotting
        save_time_elapsed_distribution(self.data.copy())

        # Filter reels that are less than 7 days old
        # To allow engagement to accumulate, we want to remove "too recent" reels.
        # We assume that reels after 7 days are receiving negligible amount of engagement.
        # These reels consist of less than 3% of the total data.
        self.data = self.data[self.data['time_elapsed'] >= pd.Timedelta(days=7)]

        print(f"Reels filtered: {self.data.shape}")
