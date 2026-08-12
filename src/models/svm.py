from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

def svm_model(X_train, y_train):

    #SVM is sensitive to feature scales
    #Standard Scaler puts the features on a simlar scale.

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",SVC())
    ])

    #Different SVm setting that GridSearchCV will test to find the best combination of hyperparameters
    param_grid = {
        "svm__C": [0.1, 1, 10], #Regularization parameter, smaller values specify stronger regularization
        "svm__kernel": ["linear", "rbf"], #Kernel type to be used in the algorithm
        "svm__gamma": ["scale", "auto"] #Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. 'scale' uses 1 / (n_features * X.var()) as value of gamma, 'auto' uses 1 / n_features
        }

    #Testing Different Combinations 
    grid_search = GridSearchCV(
        estimator = svm_pipeline,
        param_grid = param_grid,
        cv = 5,#using 5-fold cross validation to evaluate the model performance for each combination of hyperparameters
        n_jobs = -1 #using all processors to speed up the process
    )

    #Train model using training data, Find best SVM configuration.
    grid_search.fit(X_train, y_train)

    print("Best SVM Parameters: ")
    print(grid_search.best_params_)

    #returning best trained SVm pippeline
    return grid_search.best_estimator_

def predict_svm(model, X_test):
    #Use trained SVM model to make predictions
    predictions = model.predict(X_test)

    return predictions



def save_svm(model, path):
    # Save the trained SVM model so it can be reused later
    joblib.dump(model, path)

    print("SVM model saved successfully.")

def load_svm(path):
    return joblib.load(path)