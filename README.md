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
│   ├── preprocessor.py     # Data preprocessing (segmentation, merging)
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

2. **Configure the project root path**: 
   - **In the notebooks**: Open `notebooks/part1a_churn_risk_drivers.ipynb` and `notebooks/part1c_ab_test_analysis.ipynb`
     - In the first cell of each notebook, set the `project_root` variable to point to the `use_case` folder:
     ```python
     project_root = Path(r"path/to/your/use_case")
     ```
   - **In the scripts**: Open `scripts/formatter.py`
     - Update the `project_root` variable to point to the `use_case` folder
   - This ensures all imports and file paths work correctly

3. Place your data files in the `ressource/` folder:
   - `d0_behaviour_br.csv`
   - `retention_br.csv`
   - `ab_test_br.csv`

4. Run the notebooks:
   - Open `notebooks/part1a_churn_risk_drivers.ipynb` for churn analysis
   - Open `notebooks/part1c_ab_test_analysis.ipynb` for A/B test analysis

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
