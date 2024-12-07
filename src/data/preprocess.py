import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocesses the data (in place).
        """
        print("Preprocessing Data...")

        # Filter accounts
        self._filter_accounts()
        self._drop_columns()

        # Transform data
        self._transform_data()

        # Adjust timestamp timezone
        self._adjust_timestamps()

        print(f"Data preprocessed")
        return self.data

    def _filter_accounts(self) -> None:
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
        plt.hist(
            self.data["data.user_data.meta.followers_count"],
            bins=50,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        plt.xlabel("Followers")
        plt.ylabel("Frequency")
        plt.title("Distribution of Followers")
        plt.savefig("../figures/1-follower_distribution.png")
        print("Saved 1-follower_distribution.png")

        # Remove accounts with followers above the 98th percentile
        self.data = self.data[self.data['data.user_data.meta.followers_count'] <= self.data['data.user_data.meta.followers_count'].quantile(0.98)]

        # Evaluate effects
        # print(self.data["data.user_data.meta.followers_count"].describe())

        # Plot distributions in followers
        plt.hist(
            self.data["data.user_data.meta.followers_count"],
            bins=50,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        plt.xlabel("Followers")
        plt.ylabel("Frequency")
        plt.title("Distribution of Followers")
        plt.savefig('../figures/2-follower_distribution_98th.png')
        print("Saved 2-follower_distribution_98th.png")

        # Perform normal test on log followers, mild violation but good enough (p=0.018)
        print("Normality test on logged followers, p-value:", stats.normaltest(np.log1p(self.data["data.user_data.meta.followers_count"])).pvalue)

        print(f"Influncer accounts filtered: {self.data.shape}")

    def _drop_columns(self) -> None:
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

    def _transform_data(self) -> None:
        """
        Transform data:
        1. Melt data
        2. Pivot data
        3. Fix col types
        4. Rename col names
        """
        # Melt data
        melted = self.data.melt(
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
        melted[["reel_id", "field"]] = melted["variable"].str.extract(r"reel\.(\d+)\.(.+)")
        melted.sort_values(["user_id", "reel_id", "field"])

        # Pivot data
        transformed = melted.pivot(
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
            transformed[column] = transformed[column].astype(dtype)
        transformed["reel_id"] = transformed["reel_id"].astype(pd.Int64Dtype())

        # Rename index and columns
        transformed.columns.name = None
        transformed = transformed.rename(
            columns={
                "timestamp": "scraped_at",
                "taken_at_timestamp": "created_at",
                "data.user_data.meta.total_posts_count": "total_posts",
                "data.user_data.meta.followers_count": "followers",
                "data.user_data.meta.followings_count": "followings",
            }
        )

        # Assign back to data
        self.data = transformed
        print(f"Data transformed into shape: {self.data.shape}")

    def _adjust_timestamps(self) -> None:
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
    
    def _filter_reels(self) -> None:
        pass

    def _prep_cols(self) -> None:
        pass

    def _cat_days(self) -> None:
        pass

    def _cat_time(self) -> None:
        pass

    def _impute_views(self) -> None:
        pass

    def _remove_outliers(self) -> None:
        pass

    def _aggregate_reels(self) -> None:
        pass

if __name__ == "__main__":
    # Load data
    from loader import DataLoader
    data = DataLoader().get_data()

    # Preprocess data
    preprocessor = DataPreprocessor(data)
    processed_data = preprocessor.preprocess()
    print(processed_data.head())
