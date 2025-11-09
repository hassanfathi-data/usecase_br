import pandas as pd
import plotly.graph_objects as go
from scripts.statistical_analyzer import StatisticalAnalyzer


class Visualizer:
    TITLE_FONT = dict(size=14, color='black', family='Arial Black')
    
    def __init__(self, full_dataset: pd.DataFrame):
        self.df = full_dataset
    
    def qualitative_variables_impacts_on_retention(self, qualitative_vars: list, target_var: str = 'd30', min_obs_low: int = 30, min_obs_high: int = 300):
        """Create bar charts showing average retention for each qualitative variable modality."""
        analyzer = StatisticalAnalyzer(self.df)
        figures = []
        for var in qualitative_vars:
            counts = self.df[var].value_counts()
            valid_modalities = counts[counts >= min_obs_low].index
            avg = self.df[self.df[var].isin(valid_modalities)].groupby(var)[target_var].mean().sort_values(ascending=False)
            colors = ['grey' if counts[mod] < min_obs_high else 'blue' for mod in avg.index]
            p_val = analyzer.chi2_test(var, target_var)['p_value']
            sig = 'Significant' if p_val < 0.05 else 'Not Significant'
            avg_percent = avg.values * 100
            show_annotations = len(avg) <= 15
            fig = go.Figure(go.Bar(x=avg.index, y=avg_percent, marker_color=colors, 
                                  text=[f'{val:.1f}%' for val in avg_percent] if show_annotations else None,
                                  textposition='inside', textfont=dict(color='white')))
            fig.update_xaxes(type='category', categoryorder='array', categoryarray=avg.index.tolist())
            fig.update_layout(title=dict(text=f'Average {target_var} by {var} | P-Value: {p_val:.4f} ({sig})', font=self.TITLE_FONT), 
                            xaxis_title=var, yaxis_title=f'Average {target_var} (%)', height=400, showlegend=False)
            fig.show()
            figures.append(fig)
        return figures
    
    def quantitative_variable_impact_on_retention(self, target_var: str = 'd30', min_obs: int = 300):
        """Display bar charts for age and session length showing relationship with retention."""
        analyzer = StatisticalAnalyzer(self.df)
        def create_fig(var_name, labels, avg, counts, p_val):
            colors = ['grey' if counts[g] < min_obs else 'blue' for g in avg.index]
            percent = avg.values * 100
            sig = 'Significant' if p_val < 0.05 else 'Not Significant'
            fig = go.Figure(go.Bar(x=labels, y=percent, marker_color=colors, text=[f'{val:.1f}%' for val in percent], 
                                  textposition='inside', textfont=dict(color='white')))
            fig.update_layout(title=dict(text=f'{var_name} vs {target_var.upper()} Retention | P-Value: {p_val:.4f} ({sig})', font=self.TITLE_FONT), 
                            xaxis_title=var_name.lower(), yaxis_title=f'Average {target_var.upper()} Retention (%)', height=400, showlegend=False)
            fig.show()
            return fig
        age_avg = self.df.groupby('age_segmented', observed=False)[target_var].mean().sort_index()
        fig1 = create_fig('Age', [f"{int(i.left)}-{int(i.right)}" for i in age_avg.index], age_avg, 
                         self.df['age_segmented'].value_counts(), analyzer.chi2_test('age_segmented', target_var)['p_value'])
        session_avg = self.df.groupby('session_length_segmented', observed=False)[target_var].mean().sort_index()
        fig2 = create_fig('Session Length', [f"{int(i.left/60)} mins" for i in session_avg.index], session_avg,
                         self.df['session_length_segmented'].value_counts(), analyzer.chi2_test('session_length_segmented', target_var)['p_value'])
        return [fig1, fig2]
    
    def correlation_study(self, min_obs: int = 30):
        """Create heatmaps for correlation pairs with p-value and significance in title."""
        analyzer = StatisticalAnalyzer(self.df)
        pairs = [('session_length_segmented', 'had_meaningful'), ('country', 'locale'), ('country', 'timezone'), ('locale', 'timezone')]
        figures = []
        for var1, var2 in pairs:
            var1_counts = self.df[var1].value_counts()
            var2_counts = self.df[var2].value_counts()
            filtered_df = self.df[self.df[var1].isin(var1_counts[var1_counts >= min_obs].index) & 
                                  self.df[var2].isin(var2_counts[var2_counts >= min_obs].index)]
            contingency = pd.crosstab(filtered_df[var1], filtered_df[var2])
            p_val = analyzer.chi2_test_pair(var1, var2, min_obs)['p_value']
            sig = 'Significant' if p_val < 0.05 else 'Not Significant'
            convert = lambda label, v: f"{int(label.left/60)} mins" if isinstance(label, pd.Interval) and v == 'session_length_segmented' else (f"{int(label.left)}-{int(label.right)}" if isinstance(label, pd.Interval) else str(label))
            fig = go.Figure(data=go.Heatmap(z=contingency.values, x=[convert(x, var2) for x in contingency.columns], 
                                           y=[convert(y, var1) for y in contingency.index], colorscale='Blues'))
            fig.update_layout(title=dict(text=f'{var1} vs {var2} | P-Value: {p_val:.4f} ({sig})', font=self.TITLE_FONT), 
                            xaxis_title=var2, yaxis_title=var1, height=400)
            fig.show()
            figures.append(fig)
        return figures
    
    def retention_comparison(self, retention_columns: list = ['d0', 'd3', 'd7', 'd14', 'd30']) -> go.Figure:
        """Create a bar chart comparing all retention columns."""
        means_percent = [self.df[col].mean() * 100 for col in retention_columns]
        fig = go.Figure(go.Bar(x=retention_columns, y=means_percent, marker_color='blue', name='Retention',
                            text=[f'{val:.1f}%' for val in means_percent], textposition='inside', textfont=dict(color='white')))
        fig.update_layout(title=dict(text=f'Retention Comparison: {", ".join(retention_columns)}', font=self.TITLE_FONT),
            xaxis_title='Retention Period', yaxis_title='Average Retention (%)', height=500, showlegend=False)
        fig.show()
        return fig
    
    def display_distribution_histogram(self, variable: str) -> go.Figure:
        """Display histogram for a single variable with statistics."""
        data = self.df[variable]
        filtered_data = data[data <= data.quantile(0.90)]
        mean_val, median_val, max_val = data.mean(), data.median(), data.max()
        pct_zero = (data == 0).sum() / len(data) * 100
        fig = go.Figure(go.Histogram(x=filtered_data, nbinsx=167, marker_color='blue'))
        fig.update_layout(title=dict(text=f'Distribution of {variable} | Mean: {mean_val:.2f}, Median: {median_val:.2f}, Max: {max_val:.2f}, %0: {pct_zero:.2f}%', font=self.TITLE_FONT),
            xaxis_title=variable, yaxis_title='Frequency', height=400, showlegend=False)
        fig.show()
        return fig
    
    def display_distributions(self, variables: list) -> list:
        """Display histograms for specified variables."""
        return [self.display_distribution_histogram(var) for var in variables]
    
    def time_series_by_group(self, variables: list, date_column: str = 'event_date', group_column: str = 'ab_test_cohort') -> list:
        """Create time series graphs showing test and control group averages by hour for each variable."""
        self.df[date_column] = pd.to_datetime(self.df[date_column], dayfirst=True, format='mixed')
        self.df['hour'] = self.df[date_column].dt.floor('h')
        figures = []
        for var in variables:
            hourly_avg = self.df.groupby(['hour', group_column])[var].mean().reset_index()
            test_data = hourly_avg[hourly_avg[group_column] == 'test']
            control_data = hourly_avg[hourly_avg[group_column] == 'control']
            overall_test_avg = self.df[self.df[group_column] == 'test'][var].mean()
            overall_control_avg = self.df[self.df[group_column] == 'control'][var].mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_data['hour'], y=test_data[var], mode='lines+markers', 
                                    name=f'Test (Avg: {overall_test_avg:.3f})', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=control_data['hour'], y=control_data[var], mode='lines+markers', 
                                    name=f'Control (Avg: {overall_control_avg:.3f})', line=dict(color='blue')))
            fig.update_layout(title=dict(text=f'{var} Over Time by Group', font=self.TITLE_FONT),
                            xaxis_title='Time (Hour)', yaxis_title=f'Average {var}', height=400, showlegend=True)
            fig.show()
            figures.append(fig)
        return figures
    
    def roc_curves_comparison(self, roc_data: list, best_model_name: str, best_score: float) -> go.Figure:
        """Create ROC curves comparison graph."""
        fig = go.Figure([go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC = {auc:.4f})') for name, fpr, tpr, auc in roc_data] +
                       [go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash'))])
        fig.update_layout(title=dict(text=f'ROC Curves Comparison | Best Model: {best_model_name} (AUC = {best_score:.4f})', font=self.TITLE_FONT),
                         xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=500)
        fig.show()
        return fig
