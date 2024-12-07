import os
import pandas as pd
from typing import Any, Optional


class DataLoader:
    def __init__(self, path: str = "../../data/input/instagram.csv") -> None:
        self.path = self._validate(path)

    def get_data(self) -> Optional[pd.DataFrame]:
        try:
            dtypes = self._prep_dtypes()
            cols = list(dtypes.keys())

            data = pd.read_csv(
                self.path,
                encoding="utf-8",
                index_col=False,
                usecols=cols,
                dtype=dtypes,
            )
            print(f"Loaded data from {self.path}")
            return data

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def _validate(self, path: str) -> str:
        assert os.path.exists(path), f"File not found: {path}"
        assert os.path.isfile(path), f"Not a file: {path}"
        assert path.endswith(".csv"), f"Not a CSV file: {path}"
        return path

    def _prep_dtypes(self) -> dict[str, Any]:
        # Init constant var to store desired cols and their types
        dtype_dict = {
            "user_id": pd.Int64Dtype(),
            "status": pd.StringDtype(),
            "timestamp": pd.Int64Dtype(),
            "data.user_data.meta.is_private": pd.BooleanDtype(),
            "data.user_data.meta.is_verified": pd.BooleanDtype(),
            "data.user_data.meta.is_business_account": pd.BooleanDtype(),
            "data.user_data.meta.is_professional_account": pd.BooleanDtype(),
            "data.user_data.meta.has_reel": pd.BooleanDtype(),
            "data.user_data.meta.total_posts_count": pd.Int64Dtype(),
            "data.user_data.meta.followers_count": pd.Int64Dtype(),
            "data.user_data.meta.followings_count": pd.Int64Dtype(),
        }

        # Iteratively get the metrics for posts
        post_column_pattern = "data.user_data.post.{}.{}"
        post_metrics = {
            "is_video": pd.BooleanDtype(),
            "video_view_count": pd.Int64Dtype(),
            "comments_count": pd.Int64Dtype(),
            "comments_disabled": pd.BooleanDtype(),
            "taken_at_timestamp": pd.Int64Dtype(),
            "likes_count": pd.Int64Dtype(),
            "is_pinned": pd.BooleanDtype(),
        }

        # Populate dtype_dict with desired metrics for 12 posts
        for i in range(1, 13):
            for metric, dtype in post_metrics.items():
                column_name = post_column_pattern.format(i, metric)
                dtype_dict[column_name] = dtype

        # Iteratively get the metrics for reels
        reel_column_pattern = "data.user_data.reel.{}.{}"
        reel_metrics = {
            "has_audio": pd.BooleanDtype(),
            "video_view_count": pd.Int64Dtype(),
            "comments_count": pd.Int64Dtype(),
            "comments_disabled": pd.BooleanDtype(),
            "taken_at_timestamp": pd.Int64Dtype(),
            "likes_count": pd.Int64Dtype(),
            "video_duration": pd.Float64Dtype(),
        }

        # Populate dtype_dict with desired metrics for 36 reels
        for i in range(1, 37):
            for metric, dtype in reel_metrics.items():
                column_name = reel_column_pattern.format(i, metric)
                dtype_dict[column_name] = dtype

        return dtype_dict

if __name__ == "__main__":
    loader = DataLoader(path="../../data/input/instagram.csv")
    data = loader.get_data()
    print(data.head())
