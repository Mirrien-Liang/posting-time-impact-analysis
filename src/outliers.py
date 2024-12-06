## Handle Outliers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATETIME = ["scraped_at", "created_at", "created_at_pst", "scraped_at_pst"]

imputed = pd.read_csv('.././data/input/instagram_imputed.csv', index_col=False, encoding="utf-8", parse_dates=DATETIME)

imputed['time_elapsed'] = pd.to_timedelta(imputed['time_elapsed'])

# Plot distributions of engagement metrics
plt.figure(figsize=(12, 8))

# 1. Likes
plt.subplot(3, 1, 1)
plt.hist(
    imputed["likes_count"],
    bins=50,
    alpha=0.7,
    color="blue",
    edgecolor="black",
    label="Likes",
)

# 2. Comments
plt.subplot(3, 1, 2)
plt.hist(
    imputed["comments_count"],
    bins=50,
    alpha=0.7,
    color="green",
    edgecolor="black",
    label="Comments",
)

# 3. Views
plt.subplot(3, 1, 3)
plt.hist(
    imputed["video_view_count"],
    bins=50,
    alpha=0.7,
    color="red",
    edgecolor="black",
    label="Views",
)

#plt.show()
plt.savefig('.././figures/engagement_distribution.png')
print('saved engagement_distribution.png')


#DBSCAN
# Try removing reels that faked likes or comments
imputed_2 = imputed.copy()

# Make and fit pipeline
model = make_pipeline(
    StandardScaler(),
    DBSCAN(eps=3, min_samples=3),
)

# Extract labels
imputed_2['labels'] = model.fit_predict(imputed_2[['followers', 'likes_count', 'comments_count', 'video_view_count']])

# Print outliers
outliers = imputed_2[imputed_2['labels'] == -1]
# print(outliers)

# visualize outliers, likes vs views
plt.figure(figsize=(10, 6))
colors = ['red' if label == -1 else 'blue' for label in imputed_2['labels']]
plt.scatter(imputed_2['likes_count'], imputed_2['video_view_count'], c=colors, alpha=0.6, edgecolors='k')
plt.title('DBSCAN Clustering (Likes vs. Views)')
plt.xlabel('Likes')
plt.ylabel('Views')
#plt.show()
plt.savefig('.././figures/DBSCAN_likes_vs_views.png')
print('saved DBSCAN_likes_vs_views.png')

# visualize outliers, likes vs comments
plt.figure(figsize=(10, 6))
colors = ['red' if label == -1 else 'blue' for label in imputed_2['labels']]
plt.scatter(imputed_2['likes_count'], imputed_2['comments_count'], c=colors, alpha=0.6, edgecolors='k')
plt.title('DBSCAN Clustering (Likes vs. Comments)')
plt.xlabel('Likes')
plt.ylabel('Comments')
#plt.show()
plt.savefig('.././figures/DBSCAN_likes_vs_comments.png')
print('saved DBSCAN_likes_vs_comments.png')


# visualize inliers, likes vs views
plt.figure(figsize=(10, 6))
plt.scatter(imputed_2[imputed_2['labels'] != -1]['likes_count'], imputed_2[imputed_2['labels'] != -1]['video_view_count'], c='blue', alpha=0.6, edgecolors='k')
plt.title('DBSCAN Clustering (Likes vs. Views)')
plt.xlabel('Likes')
plt.ylabel('Views')
#plt.show()
plt.savefig('.././figures/DBSCAN_likes_vs_views_inliers.png')
print('saved DBSCAN_likes_vs_views_inliers.png')

# visualize inliers, likes vs comments
plt.figure(figsize=(10, 6))
plt.scatter(imputed_2[imputed_2['labels'] != -1]['likes_count'], imputed_2[imputed_2['labels'] != -1]['comments_count'], c='blue', alpha=0.6, edgecolors='k')
plt.title('DBSCAN Clustering (Likes vs. Comments)')
plt.xlabel('Likes')
plt.ylabel('Comments')
#plt.show()
plt.savefig('.././figures/DBSCAN_likes_vs_comments_inliers.png')
print('saved DBSCAN_likes_vs_comments_inliers.png')

# Remove identified outliers from data
imputed_3 = imputed_2[imputed_2["labels"] != 1]
#print(imputed_3[["followers", "likes_count", "comments_count", "video_view_count"]].describe())

# Only 19 removed, but the small number suggests additional methods needed.
# Apply sequential filtering to keep up to the 98th percentile of views,
# 99th percentile of likes, and 99th percentile of comments.
# This essentially removed about 1500 reels. The sequential filtering
# ensures that reels with extreme views are removed, but reels with extreme
# likes and comments are not unfairly over-filtered (high views usually means high engagements).
imputed_3 = imputed_3[imputed_3["video_view_count"] < imputed_3["video_view_count"].quantile(0.98)]
imputed_3 = imputed_3[imputed_3["likes_count"] < imputed_3["likes_count"].quantile(0.99)]
imputed_3 = imputed_3[imputed_3["comments_count"] < imputed_3["comments_count"].quantile(0.99)]

# Plot distributions of engagement metrics
plt.figure(figsize=(12, 8))

# 1. Likes
plt.subplot(3, 1, 1)
plt.hist(
    imputed_3["likes_count"],
    bins=50,
    alpha=0.7,
    color="blue",
    edgecolor="black",
    label="Likes",
)

# 2. Comments
plt.subplot(3, 1, 2)
plt.hist(
    imputed_3["comments_count"],
    bins=50,
    alpha=0.7,
    color="green",
    edgecolor="black",
    label="Comments",
)

# 3. Views
plt.subplot(3, 1, 3)
plt.hist(
    imputed_3["video_view_count"],
    bins=50,
    alpha=0.7,
    color="red",
    edgecolor="black",
    label="Views",
)

#plt.show()
plt.savefig('.././figures/engagement_distribution_inliers.png')
print('saved engagement_distribution_inliers.png')

imputed_3.to_csv('.././data/input/instagram_removed_outliers.csv')
#print(imputed_3[["followers", "likes_count", "comments_count", "video_view_count"]].describe())
