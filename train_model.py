import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from scipy.stats import uniform, randint
import joblib

def train_model():
    train_df = pd.read_csv("train_features.csv")
    
    X = train_df.drop(['object_id', 'target'], axis=1)
    y = train_df['target']
    
    print(f"Training on {len(X)} samples using XGBoost.")
    
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 1.0),
        'reg_lambda': uniform(1, 3.0)
    }
    
    scale_pos_weight = (len(y) - sum(y)) / sum(y)
    
    clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    
    search = RandomizedSearchCV(
        clf, 
        param_distributions=param_dist, 
        n_iter=20, 
        scoring='roc_auc', 
        cv=5, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    print("Starting Hyperparameter Tuning...")
    search.fit(X, y)
    
    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV ROC AUC: {search.best_score_:.4f}")
    
    best_clf = search.best_estimator_
    
    y_cv_prob = cross_val_predict(best_clf, X, y, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y, y_cv_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Best threshold for F1: {best_threshold:.4f}")
    
    y_pred = (y_cv_prob >= best_threshold).astype(int)
    
    print("\nCV Metrics (Approximate):")
    print(classification_report(y, y_pred))
    print(f"ROC AUC: {roc_auc_score(y, y_cv_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    best_clf.fit(X, y)
    
    joblib.dump(best_clf, "xgb_optimized.joblib")
    print("\nModel saved to xgb_optimized.joblib")

    test_df = pd.read_csv("test_features.csv")
    X_test = test_df.drop(['object_id'], axis=1)
    test_probs = best_clf.predict_proba(X_test)[:, 1]
    
    # Convert probabilities to binary predictions using the optimized threshold
    test_preds = (test_probs >= best_threshold).astype(int)
    
    submission = pd.DataFrame({
        'object_id': test_df['object_id'],
        'prediction': test_preds
    })
    
    submission.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv (Binary Labels)")

if __name__ == "__main__":
    train_model()