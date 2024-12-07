import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns



DATETIME = ["scraped_at", "created_at", "created_at_pst", "scraped_at_pst"]

data = pd.read_csv('.././data/input/instagram_removed_outliers.csv', index_col=False, encoding="utf-8", parse_dates=DATETIME)

data['time_elapsed'] = pd.to_timedelta(data['time_elapsed'])

# Aggregate by user-day-time combination
aggregated = data.groupby(
    by=['user_id', 'day_of_week_pst', 'categorized_created_at_pst'],
).agg(
    {
        'total_posts': 'mean',
        'followers': 'mean',
        'followings': 'mean',
        'likes_count': 'mean',
        'comments_count': 'mean',
        'video_view_count': 'mean',
        'video_duration': 'mean',
    }
).dropna().reset_index()

# Replace 2-col bin with 1-col bin
aggregated['day_time_group'] = aggregated['day_of_week_pst'].astype(str) + '_' + aggregated['categorized_created_at_pst'].astype(str)

# Devise engagement velocity metric
# (For each 100 views, how many likes+comments can it gain?)
# x
# (For each 10000 increase in followers, how many logged views can it gain?)
aggregated['EPF'] = np.log1p(((aggregated['likes_count'] + aggregated['comments_count']) / aggregated['followers']) * 1000)
# aggregated['EPV'] = (aggregated['likes_count'] + aggregated['comments_count']) / (np.log1p(aggregated['video_view_count'])+1) * 100

# Check EPF and EPV
plt.hist((aggregated['EPF']), bins=50)
plt.title("EPF distribution")
#plt.show()
plt.savefig('.././figures/EPF_distribution.png')
print("saved EPF_distribution.png")


# ANOVA
# Normality test
#print(stats.normaltest(aggregated['EPF']).pvalue)  # p=2.565e-77: Non Normal!
# But ANOVA is relatively robust to normality violations, particularly if the sample size is large,
# and when the group sizes are equal or similar (n > 30 per group). In addition, due to the CLT,
# with large sample sizes, the sampling distribution of the mean diffs between groups tends to be
# normal, and the impact of non-normality on the Type I error rate is smaller.

sm.qqplot(aggregated['EPF'], line='s')
plt.title("Q-Q plot of EPF")
#plt.show()
plt.savefig('.././figures/EPF_QQ.png')
print("saved EPF_QQ.png")

# Equal-Variance
# This is more critical to ANOVA, where we have p-value of 0.42: Equal Variance
print(stats.levene(*[group['EPF'].values for _, group in aggregated.groupby('day_time_group')]))

# Given the robustness and sample size, as well as the limited addressing of independency issues, we find acceptable to proceed with ANOVA.

# Perform ANOVA
groups = [group['EPF'].values for _, group in aggregated.groupby('day_time_group')]
print(stats.f_oneway(*groups))
# p-value = 2.565095870425696e-77: Significant
# There are significant differences in mean EPF across the day-time groups.


tukey = pairwise_tukeyhsd(
    endog=aggregated['EPF'],
    groups=aggregated['day_time_group'],
    alpha=0.05,
)
# print(tukey.summary())
# Specific combinations of day and time of day are associated with higher or lower engagement

# Rank pairs
# tukey.plot_simultaneous()
tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

significant_pairs = tukey_df[tukey_df['reject'] == True]
# print(significant_pairs)

significant_pairs['abs_mean_diff'] = significant_pairs['meandiff'].abs()
ranked_pairs = significant_pairs.sort_values(by='meandiff', ascending=False)
(ranked_pairs)

# Calculate mean EPF for each day-time group
group_means = aggregated.groupby('day_time_group')['EPF'].mean().reset_index()

# Sort groups by mean EPF in descending order
group_means = group_means.sort_values(by='EPF', ascending=False)
print("Average EPF for each group")
print(group_means)

tukey.plot_simultaneous()
plt.savefig(".././figures/tukey_pairs.png")
print("Saved tukey_pairs.png")

# Add predicted group means
aggregated['predicted'] = aggregated.groupby('day_time_group')['EPF'].transform('mean')


# Calculate residuals
aggregated['residuals'] = aggregated['EPF'] - aggregated['predicted']

# Plot 1: Histogram of residuals
# Approximately normally distributed
plt.figure(figsize=(8, 5))
sns.histplot(aggregated['residuals'], kde=True, bins=10, color='blue')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
#plt.show()
plt.savefig('.././figures/residuals_hist.png')
print('saved residuals_hist.png')

# Plot 2: Q-Q Plot
# The residuals appear to satisfy the normality assumption reasonably well.
# The slight deviations in the tails suggest some mild departures from normality in the extreme values.
# But we think it's not severe enough to invalidate the normality assumption.
plt.figure(figsize=(8, 5))
stats.probplot(aggregated['residuals'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
#plt.show()
plt.savefig('.././figures/residuals_QQ.png')
print("saved residuals_QQ.png")

# Plot 3: Residuals vs. Fitted Values
# Residuals are centered around 0, and the variance of residuals is relatively constant.
# No apparent violation in the equal-variance assumption, aligning with the Levene test we performed.
plt.figure(figsize=(8, 5))
sns.scatterplot(x=aggregated['predicted'], y=aggregated['residuals'])
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Fitted Values')
plt.xlabel('Fitted Values (Predicted)')
plt.ylabel('Residuals')
#plt.show()
plt.savefig('.././figures/residuals_vs_fitted.png')
print('saved residuals_vs_fitted.png')

heatmap_data = aggregated[["EPF", "categorized_created_at_pst", "day_of_week_pst"]]
heatmap_data["day_of_week_pst"] = heatmap_data["day_of_week_pst"].astype(str)
heatmap_data["categorized_created_at_pst"] = heatmap_data["categorized_created_at_pst"].astype(str)
heatmap_data = heatmap_data.pivot_table(
    values='EPF',
    index='categorized_created_at_pst',
    columns='day_of_week_pst',
    aggfunc='mean'
)

# Reorder days and times
day_order = ['1', '2', '3', '4', '5', '6', '7']
time_order = ['Morning', 'Afternoon', 'Evening']

# Reorder and ensure numeric data
heatmap_data = heatmap_data.reindex(columns=day_order).reindex(time_order)

# Fill missing values (e.g., with 0 or any placeholder value)
heatmap_data = heatmap_data.fillna(0).astype(float)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='coolwarm')

plt.title('Heatmap of Mean EPF by Day and Time')
plt.xlabel('Day of Week')
plt.ylabel('Time of Day')
plt.xticks(
    ticks=[i + 0.5 for i in range(len(day_order))],
    labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    rotation=45,
    ha='right'
)
plt.yticks(rotation=0)
plt.tight_layout()
#plt.show()
plt.savefig('.././figures/EPF_heatmap.png')
print('saved EPF_heatmap.png')

# Get top N significant pairs
top_n = 10
top_pairs = significant_pairs.head(top_n)

# Prepare data for plotting
top_pairs['Comparison'] = top_pairs['group1'] + ' vs ' + top_pairs['group2']

plt.figure(figsize=(12, 6))
sns.barplot(
    x='abs_mean_diff',
    y='Comparison',
    data=top_pairs,
    palette='Blues_d'
)

plt.title(f'Top {top_n} Significant Mean Differences in EPF')
plt.xlabel('Absolute Mean Difference in EPF')
plt.ylabel('Group Comparisons')
plt.tight_layout()
#plt.show()
plt.savefig('.././figures/EPF_diff_top.png')
print('saved EPF_diff_top.png')

# Prepare data
line_data = aggregated.copy()
day_mapping = {
    '1': 'Monday', '2': 'Tuesday', '3': 'Wednesday',
    '4': 'Thursday', '5': 'Friday', '6': 'Saturday', '7': 'Sunday'
}

# Define the custom order for the x-axis
custom_order = [
    f"{day}_{time}"
    for day in ['1', '2', '3', '4', '5', '6', '7']
    for time in ['Morning', 'Afternoon', 'Evening']
]

# Sort data by the custom order
line_data['day_time_group'] = pd.Categorical(line_data['day_time_group'], categories=custom_order, ordered=True)
line_data = line_data.sort_values('day_time_group')

# Plot the data (single continuous line)
plt.figure(figsize=(14, 6))
sns.lineplot(x='day_time_group', y='EPF', data=line_data, marker='o',)

# Customize the plot
plt.title('Mean EPF Over Days and Times (Sequential)')
plt.xlabel('Day and Time')
plt.ylabel('Mean EPF')
plt.xticks(
    ticks=range(len(custom_order)),
    labels=[
        f"{day_mapping[grp.split('_')[0]]} {grp.split('_')[1]}"
        for grp in custom_order
    ],
    rotation=45,
    ha='right'
)
plt.tight_layout()
#plt.show()
plt.savefig('.././figures/EPF_mean.png')
print('saved EPF_mean.png')