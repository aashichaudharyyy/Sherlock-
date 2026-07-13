def recommend_models(problem_type, df, target_col):
    models = []

    rows, cols = df.shape
    is_classification = "Classification" in problem_type
    imbalance = df[target_col].value_counts(normalize=True).max() if is_classification else None
    imbalanced = is_classification and imbalance is not None and imbalance > 0.75

    imbalance_note = (
        " Note: this target is imbalanced (the majority class is "
        f"{imbalance:.0%} of rows) — accuracy alone will be misleading here, "
        "so watch Precision/Recall/F1 instead."
        if imbalanced else ""
    )

    if problem_type == "Binary Classification":

        models.append({
            "Model": "Random Forest",
            "Rating": "★★★★★",
            "Reason": "Excellent baseline model. Handles mixed features and captures non-linear relationships." + imbalance_note
        })

        models.append({
            "Model": "XGBoost",
            "Rating": "★★★★★",
            "Reason": "High accuracy and performs well on structured/tabular datasets."
        })

        models.append({
            "Model": "Logistic Regression",
            "Rating": "★★★★☆",
            "Reason": "Simple, fast and highly interpretable baseline classifier."
        })

        if rows < 10000:
            models.append({
                "Model": "SVM",
                "Rating": "★★★★☆",
                "Reason": "Suitable for small to medium-sized datasets."
            })

        models.append({
            "Model": "Decision Tree",
            "Rating": "★★★☆☆",
            "Reason": "Easy to interpret but prone to overfitting."
        })

    elif problem_type == "Multi-class Classification":

        models.append({
            "Model": "Random Forest",
            "Rating": "★★★★★",
            "Reason": "Strong performance on most multiclass tabular datasets." + imbalance_note
        })

        models.append({
            "Model": "XGBoost",
            "Rating": "★★★★★",
            "Reason": "Excellent multiclass classification performance."
        })

        models.append({
            "Model": "LightGBM",
            "Rating": "★★★★☆",
            "Reason": "Very fast and efficient for larger datasets."
        })

        models.append({
            "Model": "KNN",
            "Rating": "★★★☆☆",
            "Reason": "Works well when the dataset is relatively small."
        })

        models.append({
            "Model": "Decision Tree",
            "Rating": "★★★☆☆",
            "Reason": "Simple and interpretable."
        })

    else:       # Regression

        models.append({
            "Model": "Random Forest Regressor",
            "Rating": "★★★★★",
            "Reason": "Captures complex non-linear relationships with minimal preprocessing."
        })

        models.append({
            "Model": "XGBoost Regressor",
            "Rating": "★★★★★",
            "Reason": "One of the strongest regression models for structured data."
        })

        models.append({
            "Model": "Linear Regression",
            "Rating": "★★★★☆",
            "Reason": "Fast baseline model for approximately linear relationships."
        })

        models.append({
            "Model": "Decision Tree Regressor",
            "Rating": "★★★☆☆",
            "Reason": "Easy to understand but can overfit."
        })

        if rows < 10000:
            models.append({
                "Model": "SVR",
                "Rating": "★★★☆☆",
                "Reason": "Suitable for smaller regression datasets."
            })

    return models
