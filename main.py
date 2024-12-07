import pandas as pd
import src
import warnings
from scipy.integrate import IntegrationWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=IntegrationWarning)
src.pipeline.start_pipeline()
