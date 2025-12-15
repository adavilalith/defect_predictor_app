# core/model_configs.py

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# NOTE: These imports require specific environment setup
from xgboost import XGBClassifier 
from catboost import CatBoostClassifier 

# Define configurations for each model.
# Format: 
# { 
#   'Model Name': { 
#     'class': ModelClass, 
#     'params': [ 
#       { 'name': 'param_name', 'type': 'int/float/str', 'default': value, 'range/options': value } 
#     ] 
#   } 
# }

MODEL_CONFIGS = {
    "Decision Tree": {
        "class": DecisionTreeClassifier,
        "params": [
            {'name': 'max_depth', 'type': 'int', 'default': 5, 'range': (1, 30)},
            {'name': 'min_samples_split', 'type': 'int', 'default': 2, 'range': (2, 20)},
            {'name': 'criterion', 'type': 'str', 'default': 'gini', 'options': ['gini', 'entropy']},
        ]
    },
    "Random Forest (RF)": {
        "class": RandomForestClassifier,
        "params": [
            {'name': 'n_estimators', 'type': 'int', 'default': 100, 'range': (10, 500)},
            {'name': 'max_depth', 'type': 'int', 'default': 10, 'range': (1, 50)},
            {'name': 'min_samples_leaf', 'type': 'int', 'default': 1, 'range': (1, 10)},
        ]
    },
    "Support Vector Machine (SVM)": {
        "class": SVC,
        "params": [
            {'name': 'C', 'type': 'float', 'default': 1.0, 'range': (0.1, 10.0), 'step': 0.1},
            {'name': 'kernel', 'type': 'str', 'default': 'rbf', 'options': ['linear', 'poly', 'rbf', 'sigmoid']},
            {'name': 'gamma', 'type': 'str', 'default': 'scale', 'options': ['scale', 'auto']},
        ]
    },
    "XGBoost (XGB)": {
        "class": XGBClassifier,
        "params": [
            {'name': 'n_estimators', 'type': 'int', 'default': 100, 'range': (50, 500)},
            {'name': 'max_depth', 'type': 'int', 'default': 3, 'range': (1, 15)},
            {'name': 'learning_rate', 'type': 'float', 'default': 0.1, 'range': (0.001, 0.5), 'step': 0.01},
        ]
    },
    "CatBoost": {
        "class": CatBoostClassifier,
        # CatBoost parameters are more complex, simplifying to the key ones for UI
        "params": [
            {'name': 'iterations', 'type': 'int', 'default': 100, 'range': (50, 500)},
            {'name': 'depth', 'type': 'int', 'default': 6, 'range': (1, 10)},
            {'name': 'learning_rate', 'type': 'float', 'default': 0.03, 'range': (0.001, 0.5), 'step': 0.01},
        ]
    },
}