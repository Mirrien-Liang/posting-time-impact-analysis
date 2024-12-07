import pandas as pd

from src.data.filter import DataFilter
from src.data.imputer import DataImputer
from src.data.outlier import OutlierHandler
from src.data.transformer import DataTransformer
from src.models.anova import ANOVA

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocesses the data.
        """
        print("Preprocessing Data...")

        # Filter accounts
        self.data = DataFilter(self.data).filter_accounts()

        # Transform data
        self.data = DataTransformer(self.data).transform()

        # Adjust timestamp timezone
        self.data = DataTransformer(self.data).adjust_timestamps()

        # Filter reels
        self.data = DataFilter(self.data).filter_reels()

        # Create day time categories
        self.data = DataTransformer(self.data).categorize_day_time()

        # Impute missing view counts
        self.data = DataImputer(self.data).impute_views()

        # Find outliers
        self.data = OutlierHandler(self.data).remove()

        # Aggregate reels
        self.data = DataTransformer(self.data).aggregate_reels()

        # Get EPF
        self.data = DataTransformer(self.data).get_engagement_per_follower()

        print("Data preprocessed")
        self._export_data()

        return self.data

    def _export_data(self) -> None:
        self.data.to_csv("data/processed/instagram_cleaned.csv", index=False)
        print("Processed data exported to `data/input/instagram_cleaned.csv`")

if __name__ == "__main__":
    # Load data
    from loader import DataLoader
    data = DataLoader().get_data()

    # Preprocess data
    preprocessor = DataPreprocessor(data)
    processed_data = preprocessor.preprocess()
    print(processed_data.head())
