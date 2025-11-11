import pandas as pd
import numpy as np
import plotly.graph_objects as go

TITLE_FONT = dict(size=14, color='black', family='Arial Black')
LEGEND_CONFIG = dict(yanchor="top", y=0.99, xanchor="left", x=1.01, itemsizing="constant")


def is_significant(p_val: float) -> str:
    """Return 'Significant' or 'Not Significant' based on p-value."""
    return 'Significant' if p_val < 0.05 else 'Not Significant'


def add_legend_entries(fig: go.Figure, first_item):
    """Add legend entries for color coding (only visible in legend)."""
    if len(first_item) > 0:
        fig.add_trace(go.Bar(x=[first_item[0]], y=[0], marker_color='blue',
                            name='≥ 300 observations', showlegend=True, visible='legendonly'))
        fig.add_trace(go.Bar(x=[first_item[0]], y=[0], marker_color='grey',
                            name='< 300 observations', showlegend=True, visible='legendonly'))


def convert_label(label, var_name: str) -> str:
    """Convert interval labels to readable strings."""
    if isinstance(label, pd.Interval):
        if var_name == 'session_length_segmented':
            return f"{int(label.left/60)} mins"
        return f"{int(label.left)}-{int(label.right)}"
    return str(label)


def qualitative_variables_impacts_on_retention(df: pd.DataFrame, qualitative_vars: list, target_var: str = 'd30', min_obs_low: int = 30, min_obs_high: int = 300):
    """Create bar charts showing average retention for each qualitative variable modality."""
    from scripts.statistical_analyzer import chi2_test
    figures = []
    for var in qualitative_vars:
        counts = df[var].value_counts()
        valid_modalities = counts[counts >= min_obs_low].index
        avg = df[df[var].isin(valid_modalities)].groupby(var)[target_var].mean().sort_values(ascending=False)
        colors = ['grey' if counts[mod] < min_obs_high else 'blue' for mod in avg.index]
        p_val = chi2_test(df, var, target_var)['p_value']
        sig = is_significant(p_val)
        avg_percent = avg.values * 100
        show_annotations = len(avg) <= 15

        fig = go.Figure(go.Bar(x=avg.index, y=avg_percent, marker_color=colors, 
                              text=[f'{val:.1f}%' for val in avg_percent] if show_annotations else None,
                              textposition='inside', textfont=dict(color='white'), showlegend=False))
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=avg.index.tolist())

        add_legend_entries(fig, avg.index)
        fig.update_layout(title=dict(text=f'Average {target_var} by {var} | P-Value: {p_val:.4f} ({sig})', font=TITLE_FONT), 
                        xaxis_title=var, yaxis_title=f'Average {target_var} (%)', height=400, showlegend=True,
                        legend=LEGEND_CONFIG)
        fig.show()
        figures.append(fig)
    return figures


def create_quantitative_fig(var_name, labels, avg, counts, p_val, target_var: str, min_obs: int):
    """Create bar chart for quantitative variable impact on retention."""
    colors = ['grey' if counts[g] < min_obs else 'blue' for g in avg.index]
    percent = avg.values * 100
    sig = is_significant(p_val)

    fig = go.Figure(go.Bar(x=labels, y=percent, marker_color=colors, text=[f'{val:.1f}%' for val in percent], 
                          textposition='inside', textfont=dict(color='white'), showlegend=False))
    add_legend_entries(fig, labels)
    fig.update_layout(title=dict(text=f'{var_name} vs {target_var.upper()} Retention | P-Value: {p_val:.4f} ({sig})', font=TITLE_FONT), 
                    xaxis_title=var_name.lower(), yaxis_title=f'Average {target_var.upper()} Retention (%)', height=400, showlegend=True,
                    legend=LEGEND_CONFIG)
    fig.show()
    return fig


def quantitative_variable_impact_on_retention(df: pd.DataFrame, target_var: str = 'd30', min_obs: int = 300):
    """Display bar charts for age and session length showing relationship with retention."""
    from scripts.statistical_analyzer import chi2_test
    age_avg = df.groupby('age_segmented', observed=False)[target_var].mean().sort_index()
    fig1 = create_quantitative_fig('Age', [f"{int(i.left)}-{int(i.right)}" for i in age_avg.index], age_avg, 
                     df['age_segmented'].value_counts(), chi2_test(df, 'age_segmented', target_var)['p_value'], target_var, min_obs)
    session_avg = df.groupby('session_length_segmented', observed=False)[target_var].mean().sort_index()
    fig2 = create_quantitative_fig('Session Length', [f"{int(i.left/60)} mins" for i in session_avg.index], session_avg,
                     df['session_length_segmented'].value_counts(), chi2_test(df, 'session_length_segmented', target_var)['p_value'], target_var, min_obs)
    return [fig1, fig2]


def correlation_study(df: pd.DataFrame, min_obs: int = 30):
    """Create heatmaps for correlation pairs with p-value and significance in title."""
    from scripts.statistical_analyzer import chi2_test_pair
    pairs = [('session_length_segmented', 'had_meaningful'), ('country', 'locale'), ('country', 'timezone'), ('locale', 'timezone')]
    figures = []
    for var1, var2 in pairs:
        var1_counts = df[var1].value_counts()
        var2_counts = df[var2].value_counts()
        filtered_df = df[df[var1].isin(var1_counts[var1_counts >= min_obs].index) & 
                          df[var2].isin(var2_counts[var2_counts >= min_obs].index)]
        contingency = pd.crosstab(filtered_df[var1], filtered_df[var2])
        p_val = chi2_test_pair(df, var1, var2, min_obs)['p_value']
        sig = is_significant(p_val)

        fig = go.Figure(data=go.Heatmap(z=contingency.values, x=[convert_label(x, var2) for x in contingency.columns],
                                       y=[convert_label(y, var1) for y in contingency.index], colorscale='Blues'))
        fig.update_layout(title=dict(text=f'{var1} vs {var2} | P-Value: {p_val:.4f} ({sig})', font=TITLE_FONT), 
                        xaxis_title=var2, yaxis_title=var1, height=400)
        fig.show()
        figures.append(fig)
    return figures


def retention_comparison(df: pd.DataFrame, retention_columns: list = ['d0', 'd3', 'd7', 'd14', 'd30']) -> go.Figure:
    """Create a bar chart comparing all retention columns."""
    means_percent = [df[col].mean() * 100 for col in retention_columns]
    fig = go.Figure(go.Bar(x=retention_columns, y=means_percent, marker_color='blue', name='Retention',
                        text=[f'{val:.1f}%' for val in means_percent], textposition='inside', textfont=dict(color='white')))
    fig.update_layout(title=dict(text=f'Retention Comparison: {", ".join(retention_columns)}', font=TITLE_FONT),
        xaxis_title='Retention Period', yaxis_title='Average Retention (%)', height=500, showlegend=False)
    fig.show()
    return fig


def display_distribution_histogram(df: pd.DataFrame, variable: str) -> go.Figure:
    """Display histogram for a single variable with statistics."""
    data = df[variable]
    mean_val, median_val, max_val = data.mean(), data.median(), data.max()
    pct_zero = (data == 0).sum() / len(data) * 100

    fig = go.Figure(go.Histogram(x=data, nbinsx=167, marker_color='blue'))
    fig.update_layout(title=dict(text=f'Distribution of {variable} | Mean: {mean_val:.2f}, Median: {median_val:.2f}, Max: {max_val:.2f}, %0: {pct_zero:.2f}%', font=TITLE_FONT),
        xaxis_title=variable, yaxis_title='Frequency', height=400, showlegend=False)
    fig.show()
    return fig


def display_distributions(df: pd.DataFrame, variables: list) -> list:
    """Display histograms for specified variables."""
    return [display_distribution_histogram(df, var) for var in variables]


def time_series_by_group(df: pd.DataFrame, variables: list, date_column: str = 'event_date', group_column: str = 'ab_test_cohort') -> list:
    """Create time series graphs showing test and control group averages by hour for each variable."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, format='mixed')
    df['hour'] = df[date_column].dt.floor('h')
    figures = []
    for var in variables:
        hourly_avg = df.groupby(['hour', group_column])[var].mean().reset_index()
        test_data = hourly_avg[hourly_avg[group_column] == 'test']
        control_data = hourly_avg[hourly_avg[group_column] == 'control']
        overall_test_avg = df[df[group_column] == 'test'][var].mean()
        overall_control_avg = df[df[group_column] == 'control'][var].mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_data['hour'], y=test_data[var], mode='lines+markers', 
                                name=f'Test (Avg: {overall_test_avg:.3f})', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=control_data['hour'], y=control_data[var], mode='lines+markers', 
                                name=f'Control (Avg: {overall_control_avg:.3f})', line=dict(color='blue')))
        fig.update_layout(title=dict(text=f'{var} Over Time by Group', font=TITLE_FONT),
                        xaxis_title='Time (Hour)', yaxis_title=f'Average {var}', height=400, showlegend=True)
        fig.show()
        figures.append(fig)
    return figures


def roc_curves_comparison(roc_data: list, best_model_name: str, best_score: float) -> go.Figure:
    """Create ROC curves comparison graph."""
    fig = go.Figure([go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC = {auc:.4f})') for name, fpr, tpr, auc in roc_data] +
                   [go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash'))])
    fig.update_layout(title=dict(text=f'ROC Curves Comparison | Best Model: {best_model_name} (AUC = {best_score:.4f})', font=TITLE_FONT),
                     xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=500)
    fig.show()
    return fig


def retention_correlation_heatmap(df: pd.DataFrame, retention_columns: list = ['d0', 'd3', 'd7', 'd14', 'd30']) -> go.Figure:
    """Create a heatmap showing correlations between retention variables."""
    from scripts.statistical_analyzer import retention_correlation_matrix
    corr_matrix = retention_correlation_matrix(df, retention_columns)
    if corr_matrix.empty:
        return go.Figure()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    fig.update_layout(
        title=dict(text='Retention Variables Correlation Matrix', font=TITLE_FONT),
        xaxis_title='Retention Period',
        yaxis_title='Retention Period',
        height=500,
        width=600
    )
    fig.show()
    return fig