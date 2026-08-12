from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV #to perform hyperparameter tuning, Checks all the combinations of hyperparameters and returns the best combination
import joblib #to save model to disk

#Random Forest Training

def random_forest(X_train, y_train):

    #creating random forest model, and giving random_state=42 makes the results reproducible

    rf_model = RandomForestClassifier(random_state=42)

    #giving hyperparameters to test
    #GridSearchCV will try different combinations and give best working combination
    param_grid = {
        "n_estimators": [50, 100, 200], #no.of trees in the forest
        "max_depth": [None, 5, 10], #max depth of the tree, None means nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples
        "min_samples_split": [2,5] #minimum number of samples required to split an internal node
    }

    #creating grid search object, GridSearchCV tests all parameter combinations.
    #using 5-fold cross validation to evaluate the model performance for each combination of hyperparameters
    grid_search = GridSearchCV(
        estimator = rf_model,
        param_grid = param_grid,
        cv = 5, 
        n_jobs = -1, #using all processors to speed up the process
    )

    #Train model Using training data
    grid_search.fit(X_train, y_train)

    #Displaying best parameter found by grid search
    print("Best Random Forest Parameters:")
    print(grid_search.best_params_)

    #returning best model found by grid search
    return grid_search.best_estimator_

def predict_random_forest(model, X_test):
    #making predictions using trained RF model with GridSearchCV
    predictions = model.predict(X_test)

    return predictions


def save_random_forest(model, path):
    # Save the trained model so it can be reused later
    joblib.dump(model, path)

    print("Random Forest model saved successfully.")

def load_random_forest(path):
    # Load the saved Random Forest model
    model = joblib.load(path)

    return model