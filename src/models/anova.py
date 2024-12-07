from scipy import stats
from src.visualizer.plot import save_anova_residual_analysis, save_epf_distribution, save_mean_epf_heatmap, save_mean_epf_sequential
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import pandas as pd


class ANOVA:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def perform(self) -> pd.DataFrame:
        self._check_assumptions()
        self._fit()
        self._do_post_hoc()
        self._do_residual_analysis()
        self._visualize()
        return self.data

    def _check_assumptions(self) -> None:
        save_epf_distribution(self.data)
        print(f"Logged EPF Normal Test p-value: {stats.normaltest(self.data['log_EPF']).pvalue}")  # p=2.565e-77: Non Normal!
        print(f"Logged EPF Levene Test p-value: {stats.levene(*[group['log_EPF'].values for _, group in self.data.groupby('day_time_group')]).pvalue}")

    def _fit(self) -> None:
        groups = [group['log_EPF'].values for _, group in self.data.groupby('day_time_group')]
        anova_results = stats.f_oneway(*groups)
        print(f"ANOVA p-value: {anova_results.pvalue}")
        # p-value = 4.795216128645506e-18: Significant
        # There are significant differences in mean EPF across the day-time groups.

    def _do_post_hoc(self) -> None:
        tukey = pairwise_tukeyhsd(
            endog=self.data['log_EPF'],
            groups=self.data['day_time_group'],
            alpha=0.05,
        )
        # print(tukey.summary())

        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

        significant_pairs = tukey_df[tukey_df['reject'] == True]

        significant_pairs['abs_mean_diff'] = significant_pairs['meandiff'].abs()
        # print(significant_pairs)

        ranked_pairs = significant_pairs.sort_values(by='meandiff', ascending=False)
        print("="*99)
        print(ranked_pairs.head())
        print("...")
        print("="*99)

        # Calculate mean EPF for each day-time group and sort
        group_means = self.data.groupby('day_time_group')['log_EPF'].mean().reset_index().sort_values(by='log_EPF', ascending=False)
        # print(group_means)

    def _do_residual_analysis(self) -> None:
        # Add predicted group means
        self.data['predicted'] = self.data.groupby('day_time_group')['log_EPF'].transform('mean')

        # Calculate residuals
        self.data['residuals'] = self.data['log_EPF'] - self.data['predicted']

        save_anova_residual_analysis(self.data)

    def _visualize(self) -> None:
        save_mean_epf_heatmap(self.data)
        save_mean_epf_sequential(self.data)
