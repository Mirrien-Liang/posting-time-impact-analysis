from src.data import DataLoader, DataPreprocessor
from src.models.anova import ANOVA

def start_pipeline() -> None:
    """
    Start the data pipeline
    """
    # Load data
    data = DataLoader(path="data/input/instagram.csv").get_data()

    # Preprocess data
    processed_data = DataPreprocessor(data).preprocess()

    # Apply ANOVA & Tukey HSD
    analyzed_data = ANOVA(processed_data).perform()

    analyzed_data.to_csv("data/output/instagram_analyzed.csv", index=False)

    print("Data pipeline completed successfully")
