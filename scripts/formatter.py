import pandas as pd
import os
import tempfile
import img2pdf
from pathlib import Path

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


def export_to_pdf(figures: list, filename: str = 'churn_risk_analysis.pdf'):
    """Export figures to PDF."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_PATH / filename
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = [os.path.join(tmpdir, f'fig_{i}.png') for i in range(len(figures))]
        for fig, path in zip(figures, image_paths):
            fig.write_image(path, width=800, height=400)
        with open(pdf_path, 'wb') as f:
            f.write(img2pdf.convert(image_paths))
    print(f"PDF exported to: {pdf_path}")
