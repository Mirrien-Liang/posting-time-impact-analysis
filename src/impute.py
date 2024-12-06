import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


DATETIME = ["scraped_at", "created_at", "created_at_pst", "scraped_at_pst"]


## Handle NAs
augmented = pd.read_csv('.././data/input/instagram_transform.csv', index_col=False, encoding="utf-8", parse_dates=DATETIME)

augmented['time_elapsed'] = pd.to_timedelta(augmented['time_elapsed'])


# Define bins for histogram
bin_edges = np.histogram_bin_edges(
    augmented["time_elapsed"].dt.days,
    bins=30,
)

# Histogram data for all reels
all_reels, _ = np.histogram(
    augmented["time_elapsed"].dt.days,
    bins=bin_edges,
)

# Histogram data for missing video views
missing_views, _ = np.histogram(
    augmented[augmented["video_view_count"].isna()]["time_elapsed"].dt.days,
    bins=bin_edges,
)

# Calculate proportions
proportions = (missing_views / all_reels) * 100

# Plot stacked histogram
plt.figure(figsize=(16, 8))  # Larger figure size

# Bars for all reels
plt.bar(
    bin_edges[:-1],
    all_reels,
    width=np.diff(bin_edges),
    alpha=0.75,
    color="blue",
    edgecolor="black",
    label="All Reels",
)

# Bars for missing views (stacked)
plt.bar(
    bin_edges[:-1],
    missing_views,
    width=np.diff(bin_edges),
    alpha=0.75,
    color="orange",
    edgecolor="black",
    label="Missing Views",
)

# Add percentages on top of the bars
for i, (x, total, missing) in enumerate(zip(bin_edges[:-1], all_reels, missing_views)):
    if total > 0:  # Avoid division by zero
        percentage = (missing / total) * 100
        plt.text(
            x + np.diff(bin_edges)[i] / 2,
            total + 100,  # Adjust y-position for clarity
            f"{percentage:.1f}%",
            ha="center",
            fontsize=9,
            color="black",
        )

# Add titles and labels
plt.title("Proportion of Missing Video Views Across Time Elapsed (Days)", fontsize=14)
plt.xlabel("Time Elapsed (Days)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis="y", alpha=0.5)

# Show plot
plt.tight_layout()
#plt.show()
plt.savefig(".././figures/missing_views_porportions.png")
print("saved missing_views_porportions.png")

### Impute NAs in video views with RF
#### Preprocessing data

# Preprocess data
# Get time elasped in seconds
augmented["time_elapsed_seconds"] = augmented["time_elapsed"].dt.total_seconds()

# Encode has_audio
augmented["has_audio"] = augmented["has_audio"].astype(pd.Int64Dtype())

# One-Hot Encode 'day_of_week_pst' and 'categorized_created_at_pst'
day_of_week_dummies = pd.get_dummies(
    augmented["day_of_week_pst"],
    prefix="day",
    drop_first=True,  # Use Monday as reference
).astype(pd.Int64Dtype())
time_category_dummies = pd.get_dummies(
    augmented["categorized_created_at_pst"],
    prefix="time_cat",
    drop_first=True,  # Use Evening as reference
).astype(pd.Int64Dtype())

# Concatenate the dummy variables to the dataframe
augmented = pd.concat([augmented, day_of_week_dummies, time_category_dummies], axis=1)

# Define features selected
X_COLS = [
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
Y_COL = "video_view_count"

# del day_of_week_dummies, time_category_dummies; gc.collect()

train_data = augmented[augmented["video_view_count"].notna()].copy()
missing_data = augmented[augmented["video_view_count"].isna()].copy()

# Plot distribution
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.hist(train_data[Y_COL], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of original video views')
plt.xlabel('video_view_count')
plt.ylabel('Frequency')
# plt.show()

# Highly skewed to the right
train_data['video_view_count_log'] = np.log1p(train_data[Y_COL])  # Get log(1+x) as view can be 0
Y_COL_TRANSFORMED = 'video_view_count_log'

# Plot again, satisfied
plt.subplot(1, 2, 2)
plt.hist(train_data[Y_COL_TRANSFORMED], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of video views after log transformation')
plt.xlabel('video_view_count_log')
plt.ylabel('Frequency')
#plt.show()
plt.savefig('.././figures/view_distribution.png')
print('saved view_distribution.png')

#### Examine features correlation
corr_matrix = train_data[X_COLS + [Y_COL_TRANSFORMED]].corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation with video_view_count")
#plt.show()
plt.savefig('.././figures/features_correlation.png')
print('saved features_correlation.png')

# High positive correlation found in likes_count and comments_count (0.73); Dropping comments_count;

# Medium positive correlation found in followers and followings is fine.

# Medium negative correlation found in time_cat (around -0.50) is normal due to the nature of one-hot encoding.
# It's not a concern in RF. But in case other models (e.g., linear regression), it might introduce multicollinearty issues.
# So we drop the first categories in time_cat (and day_of_week, because why not) to reduce potential multicollinearity.


#### Fit a Random Forest
# Define features and target
X = train_data[X_COLS]
y = train_data[Y_COL_TRANSFORMED]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)

# Used hyperparameters tuned by Grid and Randomized CV Search
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42,  # Make reproducible
)
rf.fit(X_train, y_train)  # Takes about 1 minutes?
print("Random forest score: " + str(rf.score(X_valid, y_valid)))  # R-squared Score: 0.8134437488337324


# Generate predictions on missing data
imputed_log_views = rf.predict(missing_data[X_COLS])

# Convert the logged views back
imputed_values = np.expm1(imputed_log_views).round().astype(int)  # Since we used p1, we use m1 here
imputed = augmented.copy()
imputed["is_view_count_imputed"] = imputed[Y_COL].isna().astype(pd.Int64Dtype())
imputed.loc[augmented[Y_COL].isna(), Y_COL] = imputed_values


# Calculate residuals
residuals = y_valid - rf.predict(X_valid)

# Scatter plot of residuals vs. predicted values
plt.figure(figsize=(18,6))
plt.subplot(1, 2, 1)
plt.scatter(rf.predict(X_valid), residuals, alpha=0.5, edgecolor='k')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')



# Distribution of residuals
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
#plt.show()
plt.savefig('.././figures/residuals_distribution.png')
print('saved residuals_distribution.png')

# The RF model appears well fitted, with residuals having no clear pattern, supporting the linearity and equal-variance assumption.

train_data['video_view_count_log'] = np.log1p(train_data[Y_COL])  # Get log(1+x) as view can be 0
Y_COL_TRANSFORMED = 'video_view_count_log'

imputed.to_csv('.././data/input/instagram_imputed.csv')