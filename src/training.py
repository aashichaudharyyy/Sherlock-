import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier

def select_model(selected_model, problem_type):

    # NOTE: keys here must exactly match the "Model" names produced in
    # recommendations.recommend_models(), otherwise the UI selectbox can hand
    # back a name this dict doesn't know about (KeyError at train time).
    classification_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    }

    regression_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "XGBoost Regressor": XGBRegressor(random_state=42),
    }

    if "Classification" in problem_type:
        return classification_models[selected_model]

    return regression_models[selected_model]

def prepare_target(y, problem_type):

    if "Classification" in problem_type:
        if not pd.api.types.is_numeric_dtype(y):
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
            return y, encoder

    return y, None
