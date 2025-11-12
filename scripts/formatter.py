import pandas as pd
import os
import tempfile
import img2pdf
from pathlib import Path
from scripts.preprocessor import remove_top_1_percent
from scripts.statistical_analyzer import student_test, permutation_test_func

# Get project root (scripts/ -> use_case/)
# IMPORTANT: Update this path to point to your use_case folder
project_root = Path(__file__).resolve().parent.parent
OUTPUT_PATH = project_root / "output"


def abtest_summary_table(df: pd.DataFrame, variables: list, group_column: str = 'ab_test_cohort') -> pd.DataFrame:
    """Create a table with means, t-test and permutation test results for each variable."""
    results = []
    for var in variables:
        df_clean = remove_top_1_percent(df, [var])
        test_group = df_clean[df_clean[group_column] == 'test']
        control_group = df_clean[df_clean[group_column] == 'control']

        student_p = student_test(df, var, group_column)
        perm_p = permutation_test_func(df, var, group_column)
        student_sig = 'is_significant' if student_p < 0.05 else 'is_not_significant'
        perm_sig = 'is_significant' if perm_p < 0.05 else 'is_not_significant'

        results.append({
            'Variable': var,
            'N Test': len(test_group),
            'N Control': len(control_group),
            'Average Test': round(test_group[var].mean(), 3),
            'Average Control': round(control_group[var].mean(), 3),
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
