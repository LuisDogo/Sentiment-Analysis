from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.linear_model import LogisticRegression


def main():
    rs = 2024
    # train different ml models. Target : f1-macro
    # train the best model with the full training data
    clf = LogisticRegression(random_state = rs)
    param_distributions = {
        "penalty": [None, "l1", "l2", "elasticnet"],
        "solver": ["saga", "lbfgs", "newton-cg", "sag"],
        "max_iter": [50, 100, 150],
        "C": [1, 1.1, 1.2],
    }
    search = HalvingRandomSearchCV(
        clf,
        param_distributions,
        cv = 5, # 5 fold splitting for training
        resource='n_estimators',
        max_resources=10,
        random_state = rs).fit(X, y)
    
    goat = search.best_params_
    return 0


if __name__ == "__main__":
    main()