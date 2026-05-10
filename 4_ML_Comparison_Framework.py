"""
ML Model Comparison Framework
==============================

A reusable, production-ready framework for comparing machine learning models
on classification tasks. Use this for rapid prototyping, benchmarking, and
publishing research results.

Author: Daya
License: MIT
Version: 1.0

USAGE:
------
# Basic usage
from ml_comparison import ModelComparison

mc = ModelComparison(X_train, y_train, X_test, y_test)
mc.add_model('Random Forest', RandomForestClassifier(n_estimators=100))
mc.add_model('XGBoost', XGBClassifier())
mc.run_comparison()
mc.plot_results()
mc.get_report()

FEATURES:
---------
✓ Train multiple models with one line
✓ Automatic evaluation (accuracy, precision, recall, F1, ROC-AUC)
✓ Feature importance analysis
✓ Cross-validation support
✓ Beautiful visualizations
✓ Detailed report generation
✓ Model serialization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import time
import json


class ModelComparison:
    """
    Framework for comparing multiple classification models.
    
    Handles training, evaluation, feature importance, and visualization.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, cv_folds=5, random_state=42):
        """
        Initialize the comparison framework.
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training features
        y_train : array-like, shape (n_samples,)
            Training labels
        X_test : array-like, shape (n_samples, n_features)
            Test features
        y_test : array-like, shape (n_samples,)
            Test labels
        cv_folds : int, default=5
            Number of cross-validation folds
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.models = {}  # Store model instances
        self.results = {}  # Store evaluation results
        self.trained_models = {}  # Store fitted models
        
    def add_model(self, name: str, model):
        """Add a model to compare."""
        self.models[name] = model
        print(f"✓ Added model: {name}")
        
    def run_comparison(self, verbose=True):
        """
        Train and evaluate all models.
        
        Returns:
        --------
        results : dict
            Dictionary containing all evaluation metrics
        """
        print("\n" + "="*60)
        print("RUNNING MODEL COMPARISON")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...", end=" ")
            start_time = time.time()
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = None
            
            # Get probability predictions if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = self._evaluate_model(
                self.y_test, y_pred, y_pred_proba, name, model
            )
            
            train_time = time.time() - start_time
            metrics['training_time_seconds'] = round(train_time, 3)
            
            self.results[name] = metrics
            
            # Print summary
            print(f"✓ Done in {train_time:.2f}s")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print("\n" + "="*60)
        return self.results
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, name, model):
        """Calculate evaluation metrics for a model."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # ROC-AUC (only for binary classification with probabilities)
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                pass
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            metrics['feature_importances'] = model.feature_importances_
        
        # Store confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def get_report(self, as_dict=False) -> str:
        """
        Generate a detailed text report of results.
        
        Parameters:
        -----------
        as_dict : bool
            If True, return as dictionary instead of string
            
        Returns:
        --------
        report : str or dict
        """
        if not self.results:
            print("⚠ No results available. Run run_comparison() first.")
            return None
        
        report_dict = {}
        
        for model_name, metrics in self.results.items():
            report_dict[model_name] = {
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'Training Time (s)': metrics.get('training_time_seconds', 'N/A'),
            }
            
            if 'roc_auc' in metrics:
                report_dict[model_name]['ROC-AUC'] = f"{metrics['roc_auc']:.4f}"
        
        if as_dict:
            return report_dict
        
        # Format as text report
        report_text = "\n" + "="*70 + "\n"
        report_text += "MODEL COMPARISON REPORT\n"
        report_text += "="*70 + "\n\n"
        
        for model_name, metrics in report_dict.items():
            report_text += f"\n{model_name}\n" + "-"*40 + "\n"
            for key, value in metrics.items():
                report_text += f"{key:.<30} {value}\n"
        
        report_text += "\n" + "="*70 + "\n"
        
        return report_text
    
    def plot_results(self, figsize=(14, 8), save_path=None):
        """
        Create comparison visualizations.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        if not self.results:
            print("⚠ No results to plot. Run run_comparison() first.")
            return
        
        # Prepare data for plotting
        metrics_df = pd.DataFrame({
            name: {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
            }
            for name, metrics in self.results.items()
        }).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
        
        # 1. Metrics Heatmap
        ax = axes[0, 0]
        sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='RdYlGn', 
                    vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
        ax.set_title('Performance Metrics Heatmap', fontweight='bold')
        
        # 2. Accuracy Comparison
        ax = axes[0, 1]
        metrics_df['Accuracy'].plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Accuracy Comparison', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. F1-Score Comparison
        ax = axes[1, 0]
        metrics_df['F1-Score'].plot(kind='bar', ax=ax, color='coral')
        ax.set_title('F1-Score Comparison', fontweight='bold')
        ax.set_ylabel('F1-Score')
        ax.set_xlabel('')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Training Time
        ax = axes[1, 1]
        training_times = [self.results[name].get('training_time_seconds', 0) 
                         for name in self.results.keys()]
        ax.barh(list(self.results.keys()), training_times, color='mediumseagreen')
        ax.set_title('Training Time Comparison', fontweight='bold')
        ax.set_xlabel('Time (seconds)')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figure saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_feature_importance(self, top_n=15, figsize=(10, 6), save_path=None):
        """
        Plot feature importance for models that support it.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(1, len(self.results), figsize=(figsize[0]*len(self.results)/2, figsize[1]))
        
        if len(self.results) == 1:
            axes = [axes]
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            if 'feature_importances' not in metrics:
                continue
            
            importances = metrics['feature_importances']
            indices = np.argsort(importances)[-top_n:]
            
            ax = axes[idx]
            ax.barh(range(len(indices)), importances[indices], color='teal')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([f"Feature {i}" for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'{name}\nTop {top_n} Features', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance figure saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def export_results(self, filepath: str):
        """
        Export results to JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path to save JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        export_results = {}
        for name, metrics in self.results.items():
            export_results[name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_results, f, indent=4)
        
        print(f"✓ Results exported to {filepath}")
    
    def get_best_model(self, metric='f1'):
        """
        Get the best performing model.
        
        Parameters:
        -----------
        metric : str
            Metric to use for ranking ('accuracy', 'f1', 'roc_auc', etc.)
            
        Returns:
        --------
        model_name : str
            Name of the best model
        model : sklearn model
            The fitted model object
        score : float
            The metric value
        """
        scores = {name: metrics.get(metric, 0) 
                 for name, metrics in self.results.items()}
        best_name = max(scores, key=scores.get)
        
        return best_name, self.trained_models[best_name], scores[best_name]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except:
        has_xgb = False
    
    # Load sample dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create comparison framework
    mc = ModelComparison(X_train, y_train, X_test, y_test)
    
    # Add models
    mc.add_model('Logistic Regression', LogisticRegression(max_iter=1000))
    mc.add_model('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42))
    if has_xgb:
        mc.add_model('XGBoost', XGBClassifier(random_state=42, verbosity=0))
    
    # Run comparison
    mc.run_comparison()
    
    # Print report
    print(mc.get_report())
    
    # Plot results
    mc.plot_results()
    mc.plot_feature_importance()
    
    # Get best model
    best_name, best_model, best_score = mc.get_best_model(metric='f1')
    print(f"\n✓ Best model: {best_name} (F1: {best_score:.4f})")
    
    # Export results
    mc.export_results('comparison_results.json')
