from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    """
    Train the Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


def predict(model, X_test):
    """
    Generate predictions using the trained model.
    """
    return model.predict(X_test)


def predict_probability(model, X_test):
    """
    Generate probability predictions using the trained model.
    """
    return model.predict_proba(X_test)