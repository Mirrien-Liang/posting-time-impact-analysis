import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats


def save_follower_distribution(data: pd.DataFrame, log: bool = False, suffix: str = "") -> None:
    """
    Plot distributions in followers
    """
    plt.hist(
        data["data.user_data.meta.followers_count"] if not log else np.log(data["data.user_data.meta.followers_count"]),
        bins=50,
        alpha=0.7,
        color="blue",
        edgecolor="black",
    )
    plt.ylabel("Frequency")
    filename = "follower_distribution"

    if log:
        plt.xlabel("Log Followers")
        plt.title("Distribution of Log Followers")
        filename += "_log"
    else:
        plt.xlabel("Followers")
        plt.title("Distribution of Followers")
    
    filename += ("_" + suffix) if suffix else ""

    plt.savefig(f"figures/{filename}.png")
    print(f"Saved {filename}.png")
    plt.close()

def save_time_elapsed_distribution(data: pd.DataFrame) -> None:
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
        "1 Day",
        "7 Days",
        "1 Month",
        "3 Months",
        "12 Months",
        "2 Years",
        "More than 2 Years",
    ]
    
    # Create a new column for bins
    data['time_elapsed_category'] = pd.cut(data['time_elapsed'].dt.days, bins=bins, labels=labels, right=False)

    # Calculate distribution and percentages
    distribution = data['time_elapsed_category'].value_counts().sort_index()
    percentages = (distribution / len(data)) * 100

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

    # Save the plot
    plt.savefig("figures/elapsed_time_distribution.png")
    print("Saved elapsed_time_distribution.png")
    plt.close()

def save_time_distribution(data: pd.DataFrame, suffix: str) -> None:
    # Plot distribution of `time_category_pst`
    time_category_counts = data['categorized_created_at_pst'].value_counts()
    time_category_percentages = (time_category_counts / len(data)) * 100

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
    plt.savefig(f"figures/time_distribution_{suffix}.png")
    print(f"Saved time_distribution_{suffix}.png")
    plt.close()

def save_day_distribution(data: pd.DataFrame) -> None:
    day_of_week_counts = data['day_of_week_pst'].value_counts().sort_index()
    day_of_week_percentages = (day_of_week_counts / len(data)) * 100

    _, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(day_of_week_counts.index, day_of_week_counts.values, tick_label=day_of_week_counts.index)

    for bar, percent in zip(bars, day_of_week_percentages):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{percent:.1f}%", 
                ha='center', va='bottom')

    ax1.set_title("Distribution of Day of Week (PST)", fontsize=16)
    ax1.set_xlabel("Day of Week (Monday=1, Sunday=7)", fontsize=14)
    ax1.set_ylabel("Count", fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/day_distribution.png")
    print("Saved day_distribution.png")
    plt.close()

def save_missing_view_over_time_elapsed(data: pd.DataFrame) -> None:
    """
    Plot missing view over time elapsed
    """
    # Define bins for histogram
    bin_edges = np.histogram_bin_edges(
        data["time_elapsed"].dt.days,
        bins=30,
    )

    # Histogram data for all reels
    all_reels, _ = np.histogram(
        data["time_elapsed"].dt.days,
        bins=bin_edges,
    )

    # Histogram data for missing video views
    missing_views, _ = np.histogram(
        data[data["video_view_count"].isna()]["time_elapsed"].dt.days,
        bins=bin_edges,
    )

    # Calculate proportions
    # proportions = (missing_views / all_reels) * 100

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
    plt.savefig("figures/missing_view_over_time_elapsed.png")
    print("Saved missing_view_over_time_elapsed.png")
    plt.close()


def save_missing_view_distribution(data: pd.DataFrame, col: str) -> None:
    plt.figure(figsize=(16, 8))

    plt.hist(data[col], bins=50, alpha=0.7, color="blue", edgecolor="black")
    plt.title("Distribution of Video Views")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.savefig(f"figures/{col}_distribution.png")
    print(f"Saved {col}_distribution.png")
    plt.close()

def save_rf_corr_matrix_heatmap(data: pd.DataFrame, features: list[str]) -> None:
    """
    Plot correlation matrix heatmap
    """
    corr_matrix = data[features].corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation with video_view_count")
    plt.savefig("figures/rf_corr_matrix_heatmap.png")
    print("Saved rf_corr_matrix_heatmap.png")
    plt.close()

def save_rf_residual_analysis(residuals: np.ndarray, predictions: np.ndarray) -> None:
    """
    Plot residual analysis
    """
    # Scatter plot of residuals vs. predicted values
    plt.figure(figsize=(18,6))
    plt.subplot(1, 2, 1)
    plt.scatter(predictions, residuals, alpha=0.5, edgecolor='k')
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
    plt.savefig("figures/rf_residual_analysis.png")
    print("Saved rf_residual_analysis.png")
    plt.close()

def save_engagement_metrics_distribution(data: pd.DataFrame, suffix: str) -> None:
    # Plot distributions of engagement metrics
    plt.figure(figsize=(12, 8))

    # 1. Likes
    plt.subplot(3, 1, 1)
    plt.hist(
        data["likes_count"],
        bins=50,
        alpha=0.7,
        color="blue",
        edgecolor="black",
        label="Likes",
    )

    # 2. Comments
    plt.subplot(3, 1, 2)
    plt.hist(
        data["comments_count"],
        bins=50,
        alpha=0.7,
        color="green",
        edgecolor="black",
        label="Comments",
    )

    # 3. Views
    plt.subplot(3, 1, 3)
    plt.hist(
        data["video_view_count"],
        bins=50,
        alpha=0.7,
        color="red",
        edgecolor="black",
        label="Views",
    )

    plt.savefig(f"figures/engagement_metrics_distribution_{suffix}.png")
    print(f"Saved engagement_metrics_distribution_{suffix}.png")
    plt.close()

def save_epf_distribution(data: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(
        data["EPF"],
        bins=20,
        alpha=0.7,
        color="red",
        edgecolor="black",
        label="EPF",
    )
    plt.title("Distribution of EPF")
    plt.xlabel("EPF")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(
        data["log_EPF"],
        bins=20,
        alpha=0.7,
        color="blue",
        edgecolor="black",
        label="Logged EPF",
    )
    plt.title("Distribution of Logged EPF")
    plt.xlabel("Logged EPF")
    plt.ylabel("Frequency")
    plt.savefig("figures/epf_distribution.png")
    print("Saved epf_distribution.png")
    plt.close()

def save_anova_residual_analysis(data: pd.DataFrame) -> None:
    # Plot 1: Histogram of residuals
    # Approximately normally distributed
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(data['residuals'], kde=True, bins=10, color='blue')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    # plt.show()

    # Plot 2: Q-Q Plot
    # The residuals appear to satisfy the normality assumption reasonably well.
    # The slight deviations in the tails suggest some mild departures from normality in the extreme values.
    # But we think it's not severe enough to invalidate the normality assumption.
    plt.subplot(1, 3, 2)
    stats.probplot(data['residuals'], dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')

    # Plot 3: Residuals vs. Fitted Values
    # Residuals are centered around 0, and the variance of residuals is relatively constant.
    # No apparent violation in the equal-variance assumption, aligning with the Levene test we performed.
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=data['predicted'], y=data['residuals'])
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs. Fitted Values')
    plt.xlabel('Fitted Values (Predicted)')
    plt.ylabel('Residuals')
    plt.savefig("figures/anova_residual_analysis.png")
    print("Saved anova_residual_analysis.png")
    plt.close()

def save_mean_epf_heatmap(data: pd.DataFrame) -> None:
    heatmap_data = data[["log_EPF", "categorized_created_at_pst", "day_of_week_pst"]]
    heatmap_data["day_of_week_pst"] = heatmap_data["day_of_week_pst"].astype(str)
    heatmap_data["categorized_created_at_pst"] = heatmap_data["categorized_created_at_pst"].astype(str)
    heatmap_data = heatmap_data.pivot_table(
        values='log_EPF',
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

    plt.title('Heatmap of Mean Logged EPF by Day and Time')
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
    plt.savefig("figures/mean_epf_heatmap.png")
    print("Saved mean_epf_heatmap.png")
    plt.close()

def save_mean_epf_sequential(data: pd.DataFrame) -> None:
    # Prepare data
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
    data['day_time_group'] = pd.Categorical(data['day_time_group'], categories=custom_order, ordered=True)
    data = data.sort_values('day_time_group')

    # Plot the data (single continuous line)
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='day_time_group', y='log_EPF', data=data, marker='o',)

    # Customize the plot
    plt.title('Mean Logged EPF Over Days and Times (Sequential)')
    plt.xlabel('Day and Time')
    plt.ylabel('Mean Logged EPF')
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
    plt.savefig("figures/mean_epf_sequential.png")
    print("Saved mean_epf_sequential.png")
    plt.close()
