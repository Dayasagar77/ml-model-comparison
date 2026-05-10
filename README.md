# ml-model-comparison
Automated ML model comparison framework with feature importance analysis
A production-ready Python framework for comparing multiple machine learning classification models. Automatically trains, evaluates, and compares models with detailed metrics and visualizations.

## Features

✓ Train multiple models with one line of code  
✓ Automatic evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)  
✓ Feature importance analysis  
✓ Cross-validation support  
✓ Beautiful comparison visualizations  
✓ Detailed report generation  
✓ Model serialization & export  

## Installation

```bash
pip install scikit-learn xgboost pandas numpy matplotlib seaborn
```

## Quick Start

```python
from ml_comparison import ModelComparison
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Initialize framework
mc = ModelComparison(X_train, y_train, X_test, y_test)

# Add models to compare
mc.add_model('Logistic Regression', LogisticRegression())
mc.add_model('Random Forest', RandomForestClassifier())

# Run comparison
mc.run_comparison()

# View results
print(mc.get_report())
mc.plot_results()

# Get best model
best_name, best_model, score = mc.get_best_model(metric='f1')
```

## Output

The framework generates:
- **Performance metrics** (accuracy, precision, recall, F1, ROC-AUC)
- **Confusion matrices** for each model
- **Feature importance rankings**
- **Comparison visualizations** (heatmaps, bar charts)
- **Text report** with all results
- **JSON export** for further analysis

## Supported Models

Works with any scikit-learn compatible model:
- Logistic Regression
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machines (SVM)
- Neural Networks
- And many more...

## Example Usage

See `4_ML_Comparison_Framework.py` for a complete example with the Breast Cancer dataset.

## Author

Daya S.  
Technical Developer | Python Specialist  
https://www.upwork.com/freelancers/~01f2c4cfeeba044788

## License

MIT License - Feel free to use and modify for your projects.
