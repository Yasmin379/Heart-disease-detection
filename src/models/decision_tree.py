from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train):
    """
    Train and tune the Decision Tree model using GridSearchCV.
    """

    model = DecisionTreeClassifier(random_state=42)

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    return grid_search.best_estimator_


def predict(model, X_test):
    """
    Generate predictions using the trained Decision Tree.
    """
    return model.predict(X_test)