import pandas as pd

from src.models.dbscan import DBSCAN
from src.visualizer.plot import save_engagement_metrics_distribution

class OutlierHandler:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def remove(self) -> None:
        save_engagement_metrics_distribution(self.data, "pre_outlier")
        self._fit_dbscan()
        self._sequential_filtering()
        save_engagement_metrics_distribution(self.data, "post_outlier")
        print(f"Outliers removed: {self.data.shape}")
        return self.data

    def _fit_dbscan(self) -> None:
        self.data = DBSCAN(self.data).remove_outliers()

    def _sequential_filtering(self) -> None:
        """
        Only 19 removed, but the small number suggests additional methods needed.
        Apply sequential filtering to keep up to the 98th percentile of views,
        99th percentile of likes, and 99th percentile of comments.
        This essentially removed about 1500 reels. The sequential filtering
        ensures that reels with extreme views are removed, but reels with extreme
        likes and comments are not unfairly over-filtered (high views usually means high engagements).
        """
        self.data = self.data[self.data["video_view_count"] < self.data["video_view_count"].quantile(0.98)]
        self.data = self.data[self.data["likes_count"] < self.data["likes_count"].quantile(0.99)]
        self.data = self.data[self.data["comments_count"] < self.data["comments_count"].quantile(0.99)]
