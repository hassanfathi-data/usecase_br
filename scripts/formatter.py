import pandas as pd
import os
import tempfile
from scripts.statistical_analyzer import StatisticalAnalyzer
import img2pdf

OUTPUT_PATH = "C:/Users/as_cu/Desktop/use_case/output"


class Formatter:
    def __init__(self, dataset: pd.DataFrame):
        self.df = dataset
        self.statistical_analyzer = StatisticalAnalyzer(dataset)
    
    def chi2_tests_table(self, variables: list, target_var: str = 'd30') -> pd.DataFrame:
        """Format chi-square test results into a table with significance column."""
        df = self.statistical_analyzer.chi2_test_multiple(variables, target_var)
        df['Is Significant'] = (df['Chi2 P-Value'] < 0.05).map({True: 'Yes', False: 'No'})
        return df
    
    def correlation_tests_table(self) -> pd.DataFrame:
        """Format correlation test results into a table with significance column."""
        df = self.statistical_analyzer.correlation_tests()
        df['Is Significant'] = (df['P-Value'] < 0.05).map({True: 'Yes', False: 'No'})
        return df
    
    def abtest_summary_table(self, variables: list, group_column: str = 'ab_test_cohort') -> pd.DataFrame:
        """Create a table with means, t-test and permutation test results for each variable."""
        from scripts.preprocessor import PreProcessor
        preprocessor = PreProcessor()
        results = []
        for var in variables:
            df_without_outlier = preprocessor.remove_top_1_percent(self.df, [var])
            n_test = len(df_without_outlier[df_without_outlier[group_column] == 'test'])
            n_control = len(df_without_outlier[df_without_outlier[group_column] == 'control'])
            avg_test = round(df_without_outlier[df_without_outlier[group_column] == 'test'][var].mean(), 3)
            avg_control = round(df_without_outlier[df_without_outlier[group_column] == 'control'][var].mean(), 3)
            student_p = self.statistical_analyzer.student_test(var, group_column)
            perm_p = self.statistical_analyzer.permutation_test(var, group_column)
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
    
    def export_to_pdf(self, figures: list, filename: str = 'churn_risk_analysis.pdf'):
        """Export figures to PDF."""
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        pdf_path = os.path.join(OUTPUT_PATH, filename)
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = [os.path.join(tmpdir, f'fig_{i}.png') for i in range(len(figures))]
            for fig, path in zip(figures, image_paths):
                fig.write_image(path, width=800, height=400)
            with open(pdf_path, 'wb') as f:
                f.write(img2pdf.convert(image_paths))
        print(f"PDF exported to: {pdf_path}")

