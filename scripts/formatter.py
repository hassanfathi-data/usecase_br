import pandas as pd
import os
import tempfile
import img2pdf
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get project root (scripts/ -> use_case/)
project_root = Path(r"C:\Users\as_cu\Desktop\use_case")
OUTPUT_PATH = project_root / "output"


def abtest_summary_table(df: pd.DataFrame, variables: list, group_column: str = 'ab_test_cohort') -> pd.DataFrame:
    """Create a table with means, t-test and permutation test results for each variable."""
    from scripts.preprocessor import remove_top_1_percent
    from scripts.statistical_analyzer import student_test, permutation_test_func
    results = []
    for var in variables:
        df_without_outlier = remove_top_1_percent(df, [var])
        n_test = len(df_without_outlier[df_without_outlier[group_column] == 'test'])
        n_control = len(df_without_outlier[df_without_outlier[group_column] == 'control'])
        avg_test = round(df_without_outlier[df_without_outlier[group_column] == 'test'][var].mean(), 3)
        avg_control = round(df_without_outlier[df_without_outlier[group_column] == 'control'][var].mean(), 3)
        student_p = student_test(df, var, group_column)
        perm_p = permutation_test_func(df, var, group_column)
        student_sig = 'is_significant' if student_p < 0.05 else 'is_not_significant'
        perm_sig = 'is_significant' if perm_p < 0.05 else 'is_not_significant'
        results.append({
            'Variable': var,
            'N Test': n_test,
            'N Control': n_control,
            'Average Test': avg_test,
            'Average Control': avg_control,
            'T-Test Result': f'p-value: {round(student_p, 3)}, {student_sig}',
            'Permutation Test Result': f'p-value: {round(perm_p, 3)}, {perm_sig}'
        })
    return pd.DataFrame(results)


def markdown_to_image(markdown_text: str, width: int = 800, height: int = None) -> str:
    """Convert markdown text to an image file path."""
    # Create figure with text
    fig = plt.figure(figsize=(width/100, height/100 if height else 10), facecolor='white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Process markdown text (remove markdown syntax for display)
    # Convert markdown headers to bold
    text = markdown_text.replace('## ', '').replace('### ', '')
    text = text.replace('**', '')
    text = text.replace('> ', '')
    
    # Wrap text
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left',
            wrap=True, family='sans-serif')
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        fig.savefig(tmp.name, bbox_inches='tight', dpi=100, facecolor='white')
        plt.close(fig)
        return tmp.name


def dataframe_to_image(df: pd.DataFrame, title: str = '', width: int = 800, height: int = None) -> str:
    """Convert a pandas DataFrame to an image file path."""
    if df.empty:
        return None
    
    # Create figure
    fig_height = height/100 if height else max(6, len(df) * 0.5 + 2)
    fig = plt.figure(figsize=(width/100, fig_height), facecolor='white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        fig.savefig(tmp.name, bbox_inches='tight', dpi=100, facecolor='white')
        plt.close(fig)
        return tmp.name


def dataframe_to_plotly_fig(df: pd.DataFrame, title: str = '') -> go.Figure:
    """Convert a pandas DataFrame to a Plotly figure."""
    if df.empty:
        return go.Figure()
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='palegreen',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[df[col].tolist() for col in df.columns],
            fill_color='white',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='black')),
        height=max(400, len(df) * 50 + 100),
        width=800
    )
    return fig


def export_to_pdf(figures: list = None, markdowns: list = None, tables: list = None, 
                  filename: str = 'churn_risk_analysis.pdf'):
    """
    Export figures, markdowns, and tables to PDF.
    
    Args:
        figures: List of Plotly figures
        markdowns: List of markdown text strings
        tables: List of pandas DataFrames or tuples of (DataFrame, title)
        filename: Output PDF filename
    """
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_PATH / filename
    
    figures = figures or []
    markdowns = markdowns or []
    tables = tables or []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        
        # Convert markdowns to images
        for i, markdown_text in enumerate(markdowns):
            img_path = markdown_to_image(markdown_text, width=800)
            image_paths.append(img_path)
        
        # Convert tables to images
        for i, table_item in enumerate(tables):
            if isinstance(table_item, tuple):
                df, title = table_item
            else:
                df, title = table_item, ''
            
            # Use plotly for better table rendering
            if isinstance(df, pd.DataFrame):
                fig = dataframe_to_plotly_fig(df, title)
                img_path = os.path.join(tmpdir, f'table_{i}.png')
                fig.write_image(img_path, width=800, height=max(400, len(df) * 50 + 100))
                image_paths.append(img_path)
        
        # Convert Plotly figures to images
        for i, fig in enumerate(figures):
            img_path = os.path.join(tmpdir, f'fig_{i}.png')
            fig.write_image(img_path, width=800, height=400)
            image_paths.append(img_path)
        
        # Create PDF
        if image_paths:
            with open(pdf_path, 'wb') as f:
                f.write(img2pdf.convert(image_paths))
            print(f"PDF exported to: {pdf_path}")
        else:
            print("No content to export to PDF")
