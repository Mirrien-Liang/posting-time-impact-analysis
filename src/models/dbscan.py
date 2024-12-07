from sklearn.cluster import DBSCAN as DBS
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class DBSCAN:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.outlier: pd.DataFrame = None
        self.model = DBS(eps=3, min_samples=3)  # Adopted from finetune

    def remove_outliers(self) -> pd.DataFrame:
        self._fit_predict()
        self._remove_outliers()
        # print(f"DBSCAN Detected Outliers: {self.outliers}")
        print(f"DBSCAN Removed Outliers: {len(self.outliers)}")
        return self.data

    def _fit_predict(self) -> None:
        ppl = make_pipeline(
            StandardScaler(),
            self.model,
        )
        labels = ppl.fit_predict(self.data[['followers', 'likes_count', 'comments_count', 'video_view_count']])

        # Add labels to the dataframe
        self.data['labels'] = labels

        # Get outliers
        self.outliers = self.data[self.data['labels'] == -1]
        
    def _remove_outliers(self) -> None:
        self.data = self.data[self.data['labels'] != -1]

    def finetune(self) -> None:
        eps_values = np.linspace(0.5, 3.0, 6)  # 0.5 to 3.0, step 0.5
        min_samples_values = range(2, 11)       # 2 to 10

        # Initialize variables to store the best parameters and score
        best_eps = None
        best_min_samples = None
        best_score = -np.inf  # Start with a very low score

        scaled_data = StandardScaler().fit_transform(self.data[['followers', 'likes_count', 'comments_count', 'video_view_count']])

        for eps in eps_values:
            for min_samples in min_samples_values:
                print(f"Trying eps={eps}, min_samples={min_samples}")
                # Fit DBSCAN
                dbscan = DBS(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(scaled_data)

                # Ignore models that classify all points as noise
                if len(set(labels)) <= 1:  # Only one cluster or all noise
                    continue

                # Evaluate the clustering
                try:
                    # Use Silhouette Score or Calinski-Harabasz Index
                    score = silhouette_score(scaled_data, labels)
                    # score = calinski_harabasz_score(scaled_data, labels)  # Optional
                except ValueError:  # Handle edge cases (e.g., single cluster)
                    continue

                print(f"Silhouette Score: {score}")

                # Update best parameters if the score is improved
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples

        # Print best parameters
        print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}, Best score: {best_score}")
        # Found eps = 3, min_samples = 3, score = 0.98

        # Assign the best model
        self.model = DBSCAN(eps=best_eps, min_samples=best_min_samples)
