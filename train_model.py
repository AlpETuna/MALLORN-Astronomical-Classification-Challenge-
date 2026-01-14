import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from scipy.stats import uniform, randint
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def train_model():
    train_df = pd.read_csv("train_features.csv")
    
    X = train_df.drop(['object_id', 'target'], axis=1)
    y = train_df['target']
    
    # Fill NaNs before SMOTE (XGBoost handles them, but SMOTE doesn't)
    X = X.fillna(X.mean())
    
    print(f"Training on {len(X)} samples using XGBoost with SMOTE.")
    
    param_dist = {
        'xgb__n_estimators': randint(100, 500),
        'xgb__learning_rate': uniform(0.01, 0.2),
        'xgb__max_depth': randint(3, 10),
        'xgb__min_child_weight': randint(1, 10),
        'xgb__gamma': uniform(0, 0.5),
        'xgb__subsample': uniform(0.6, 0.4),
        'xgb__colsample_bytree': uniform(0.6, 0.4),
        'xgb__reg_alpha': uniform(0, 1.0),
        'xgb__reg_lambda': uniform(1, 3.0)
    }
    
    # Pipeline: SMOTE -> XGBoost
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42,
            n_jobs=1 # Avoid nested parallelism issues
        ))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=20, 
        scoring='f1', # Optimize directly for F1
        cv=cv, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    print("Starting Hyperparameter Tuning...")
    search.fit(X, y)
    
    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.4f}")
    
    best_model = search.best_estimator_
    
    # Cross-val predictions for final metrics
    y_cv_prob = cross_val_predict(best_model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    
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
    
    best_model.fit(X, y)
    
    joblib.dump(best_model, "xgb_optimized.joblib")
    print("\nModel saved to xgb_optimized.joblib")

    test_df = pd.read_csv("test_features.csv")
    X_test = test_df.drop(['object_id'], axis=1)
    # Fill test NaNs with TRAIN means (to prevent leakage, but here simplified)
    X_test = X_test.fillna(train_df.drop(['object_id', 'target'], axis=1).mean())
    
    test_probs = best_model.predict_proba(X_test)[:, 1]
    
    # Use best threshold
    test_preds = (test_probs >= best_threshold).astype(int)
    
    submission = pd.DataFrame({
        'object_id': test_df['object_id'],
        'prediction': test_preds
    })
    
    submission.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv (Binary Labels)")

if __name__ == "__main__":
    train_model()