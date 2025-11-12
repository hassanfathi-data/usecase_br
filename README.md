# Use Case: Churn Risk Analysis & A/B Test Analysis

This project contains data analysis tools for:
- **Churn Risk Drivers Analysis**: Identifying factors that influence user retention
- **A/B Test Analysis**: Statistical comparison of test and control groups

## Project Structure

```
use_case/
├── notebooks/              # Jupyter notebooks for analysis
│   ├── part1a_churn_risk_drivers.ipynb
│   └── part1c_ab_test_analysis.ipynb
├── scripts/                # Python modules
│   ├── data_cleaner.py     # Data cleaning operations
│   ├── preprocessor.py     # Data preprocessing (segmentation)
│   ├── visualizer.py       # Plotly visualizations
│   ├── statistical_analyzer.py  # Statistical tests (chi2, t-test, permutation)
│   ├── formatter.py        # Table formatting and PDF export
│   ├── feature_importance.py  # ML models for feature importance
│   └── models/             # ML model implementations
│       ├── xgboost_model.py
│       ├── random_forest_model.py
│       ├── logistic_regression_model.py
│       └── model_parent.py
├── ressource/              # Data files (CSV)
├── output/                 # Generated PDF reports
└── requirements.txt        # Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Data are already available in the `ressource/` folder.

3. Run the notebooks:
   - Open `notebooks/part1a_churn_risk_drivers.ipynb` for churn analysis
   - Open `notebooks/part1c_ab_test_analysis.ipynb` for A/B test analysis

**Note:** The project paths are automatically detected based on the file locations. If this doesn't work, you can manually set the `project_root` variable:
- In the notebooks: Update the first cell in `notebooks/part1a_churn_risk_drivers.ipynb` and `notebooks/part1c_ab_test_analysis.ipynb`
- In the scripts: Update the `project_root` variable in `scripts/formatter.py` to manage the output path

## Features

### Churn Risk Analysis
- Qualitative and quantitative variable impact on retention
- Chi-square tests for significance
- Correlation heatmaps
- Feature importance using ML models (XGBoost, Random Forest, Logistic Regression)
- ROC AUC curve comparison
- PDF export of all visualizations

### A/B Test Analysis
- Distribution analysis (histograms)
- Time series analysis by group (test vs control)
- Student's t-test and permutation test
- Summary tables with p-values and significance
- PDF export of all visualizations

## Output

All generated PDFs are saved in the `output/` folder:
- `churn_risk_analysis.pdf`
- `ab_test_analysis.pdf`
