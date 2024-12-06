import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# Init constant var to store desired cols and their types
DTYPE_DICT = {
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

# Populate DTYPE_DICT with desired metrics for 12 posts
for i in range(1, 13):
    for metric, dtype in post_metrics.items():
        column_name = post_column_pattern.format(i, metric)
        DTYPE_DICT[column_name] = dtype

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
        
# Populate DTYPE_DICT with desired metrics for 36 reels
for i in range(1, 37):
    for metric, dtype in reel_metrics.items():
        column_name = reel_column_pattern.format(i, metric)
        DTYPE_DICT[column_name] = dtype

# Get a list of desired columns (for `read_csv()`)
columns_to_read = list(DTYPE_DICT.keys())

data = pd.read_csv(
    ".././data/input/instagram.csv",
    encoding="utf-8",
    index_col=False,
    usecols=columns_to_read,
    dtype=DTYPE_DICT,
)


'''## Filter Data Based on Account

Filter rows:
1. Failed jobs
2. Account with no reels
3. Private accounts
4. Accounts with total posts < 12
5. Accounts with followers < ? (50 for now)

Remove columns:
1. Post-related columns
2. Account type flags
3. Other irrelevant columns'''
# Filter out the following rows
data = data.loc[
    (data['status'] == 'successful') &
    (data['data.user_data.meta.has_reel'] == True) &
    (data['data.user_data.meta.is_private'] == False) &
    (data['data.user_data.meta.total_posts_count'] >= 12) &
    (data['data.user_data.meta.followers_count'] > 50)
]

# Filter out columns with names containing "data.user_data.post"
columns_to_drop = [col for col in data.columns if 'data.user_data.post' in col]

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

data.drop(columns=columns_to_drop, inplace=True)

# Plot distributions in followers
plt.hist(
    data["data.user_data.meta.followers_count"],
    bins=50,
    alpha=0.7,
    color="blue",
    edgecolor="black",
)
plt.xlabel("Followers")
plt.ylabel("Frequency")
plt.title("Distribution of Followers")
#plt.show()
plt.savefig('.././figures/follower_distribution.png')
print("saved follower_distribution.png")

# Looking at the result from the histogram
# Keep only the 98th quantile
data = data[data['data.user_data.meta.followers_count'] <= data['data.user_data.meta.followers_count'].quantile(0.98)]


plt.hist(
    data["data.user_data.meta.followers_count"],
    bins=50,
    alpha=0.7,
    color="blue",
    edgecolor="black",
)
plt.xlabel("Followers")
plt.ylabel("Frequency")
plt.title("Distribution of Followers")
#plt.show()
plt.savefig('.././figures/follower_distribution_98th.png')
print("saved follower_distribution_98th.png")

# Log followers
# data["log_followers"] = np.log1p(data["data.user_data.meta.followers_count"])
plt.hist(
    np.log1p(data["data.user_data.meta.followers_count"]),
    bins=50,
    alpha=0.7,
    color="blue",
    edgecolor="black",
)
plt.xlabel("Log Followers")
plt.ylabel("Frequency")
plt.title("Distribution of Log Followers")
#plt.show()
plt.savefig('.././figures/follower_distribution_log.png')
print("saved follower_distribution_log.png")

#stats.normaltest(np.log1p(data["data.user_data.meta.followers_count"])).pvalue # 0.018


### Transform data

melted = data.melt(
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

# Pivot table
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

# Convert epoch timestamp to datetime objects
transformed['created_at_timestamp'] = transformed["created_at"]
transformed['created_at'] = pd.to_datetime(transformed['created_at'], unit='s', utc=True)
transformed['created_at_pst'] = transformed['created_at'].dt.tz_convert('America/Los_Angeles')
transformed['scraped_at_timestamp'] = transformed["scraped_at"]
transformed['scraped_at'] = pd.to_datetime(transformed['scraped_at'], unit='s', utc=True)
transformed['scraped_at_pst'] = transformed['scraped_at'].dt.tz_convert('America/Los_Angeles')



#Filter
# Drop rows with N/A created_at timestamp
cleaned = transformed.dropna(subset=["created_at"])

# Drop rows with comment disabled and remove this unary column
cleaned = cleaned[cleaned["comments_disabled"] == False].drop(columns=["comments_disabled"])

# Drop rows with video duration < 1 or is NA or is > 90 seconds
cleaned = cleaned[(cleaned["video_duration"].between(1, 90, inclusive="both")) & (cleaned["video_duration"]).notna()]


## Prepare columns

### 1. Time elapsed from posted time to scraped time (in seconds)
augmented = cleaned.copy()


# Calculate the time difference between scraped_at and created_at
augmented["time_elapsed"] = augmented["scraped_at"] - augmented["created_at"]


# Define the bin edges (in days) and labels
bins = [
    0, 
    1,             # 1 Day
    7,             # 7 Days
    30,            # 1 Month
    90,            # 3 Months
    365,           # 12 Months
    730,           # 2 Years
    float('inf')   # More than 2 Years
]
labels = [
    '1 Day', '7 Days', '1 Month', '3 Months', '12 Months', '2 Years', 'More than 2 Years',
]

# Create a new column for bins
augmented['time_elapsed_category'] = pd.cut(augmented['time_elapsed'].dt.days, bins=bins, labels=labels, right=False)

# Calculate distribution and percentages
distribution = augmented['time_elapsed_category'].value_counts().sort_index()
percentages = (distribution / len(augmented)) * 100

# Plot histogram
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(distribution.index.astype(str), distribution.values)

# Add percentage labels to each bar
for bar, percent in zip(bars, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f"{percent:.1f}%", ha='center', va='bottom')

# Customize plot
ax.set_title("Distribution of Time Elapsed", fontsize=16)
ax.set_xlabel("Time Categories", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(".././figures/time_categories.png")
print("saved time_categories.png")
# Show the plot
#plt.show()

#Decision from the plot
#Take out reels that are less than 7 days old
augmented = augmented[augmented['time_elapsed'] >= pd.Timedelta(days=7)]

###Categorize created_at into weekday
# Create day_of_week column (Monday=1, ..., Sunday=7) and converted to Int64Dtype
augmented = augmented.copy()
augmented['day_of_week_pst'] = augmented['created_at_pst'].dt.dayofweek + 1  # Adjust from 0-6 to 1-7
augmented['day_of_week_pst'] = augmented['day_of_week_pst'].astype(pd.Int64Dtype())

# Use cut to categorize time in PST
# Morning (4, 11]: 05:00 to 11:59
# Afternoon (11, 16]: 12:00 to 16:59
# Evening (16, 11]: 17:00 to 23:59
# Night (-1, 4]: 00:00 AM to 04:59
augmented['categorized_created_at_pst'] = pd.cut(
    augmented['created_at_pst'].dt.hour,
    bins=[-1, 4, 11, 16, 23],
    labels=['Night', 'Morning', 'Afternoon', 'Evening']
)


# Plot distribution of `day_of_week_pst`
day_of_week_counts = augmented['day_of_week_pst'].value_counts().sort_index()
day_of_week_percentages = (day_of_week_counts / len(augmented)) * 100

fig, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.bar(day_of_week_counts.index, day_of_week_counts.values, tick_label=day_of_week_counts.index)

for bar, percent in zip(bars, day_of_week_percentages):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{percent:.1f}%", 
             ha='center', va='bottom')

ax1.set_title("Distribution of Day of Week (PST)", fontsize=16)
ax1.set_xlabel("Day of Week (Monday=1, Sunday=7)", fontsize=14)
ax1.set_ylabel("Count", fontsize=14)
plt.tight_layout()
#plt.show()
plt.savefig(".././figures/day_distribution.png")
print("saved day_distribution.png")

# Plot distribution of `time_category_pst`
time_category_counts = augmented['categorized_created_at_pst'].value_counts()
time_category_percentages = (time_category_counts / len(augmented)) * 100

fig, ax2 = plt.subplots(figsize=(10, 6))
bars = ax2.bar(time_category_counts.index, time_category_counts.values, 
               tick_label=time_category_counts.index)

for bar, percent in zip(bars, time_category_percentages):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{percent:.1f}%", 
             ha='center', va='bottom')

ax2.set_title("Distribution of Time Category (PST)", fontsize=16)
ax2.set_xlabel("Time Category", fontsize=14)
ax2.set_ylabel("Count", fontsize=14)
plt.tight_layout()
#plt.show()
plt.savefig(".././figures/time_distribution.png")

# Since Night category has only 2.2% records, we combine Evening and Night into Evening category (standing for the time after normal office hours)
augmented = augmented.replace('Night', 'Evening')

#Saved transformed data
augmented.to_csv(".././data/input/instagram_transform.csv", index=False)