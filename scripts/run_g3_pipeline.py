import os
import dill
import json
import argparse
from sklearn.model_selection import KFold, cross_validate
try:
    from src.features.engineer import engineer_features
except Exception:
    engineer_features = None
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None  # optional

try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:
    CatBoostRegressor = None  # optional


def load_cleaned_dataframe() -> pd.DataFrame:
    candidates = [
        Path('notebook/data/cleaned_student_data.csv'),
        Path('data/cleaned_student_data.csv'),
        Path('../notebook/data/cleaned_student_data.csv'),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError('cleaned_student_data.csv not found in expected locations')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="Run KFold CV in addition to holdout")
    args = parser.parse_args()

    df = load_cleaned_dataframe()
    if engineer_features is not None:
        df = engineer_features(df)

    target = 'G3'
    drop_cols = [c for c in ['Total_Score', 'Average_Score'] if c in df.columns]
    X = df.drop(drop_cols + [target], axis=1)
    y = df[target]

    cat_cols = [c for c in X.columns if X[c].dtype == 'O']
    num_cols = [c for c in X.columns if X[c].dtype != 'O']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_mean=True, with_std=True), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ],
        remainder='drop'
    )

    models = {
        'Linear_Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'KNN': KNeighborsRegressor(),
        'Decision_Tree': DecisionTreeRegressor(random_state=42),
        # Tuned models will be handled via GridSearchCV below
        'Random_Forest': RandomForestRegressor(random_state=42),
    }
    if XGBRegressor is not None:
        models['XGBRegressor'] = XGBRegressor(
            random_state=42, n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1, verbosity=0
        )
    if CatBoostRegressor is not None:
        models['CatBoost'] = CatBoostRegressor(verbose=False, random_state=42)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    rows = []
    trained = {}
    for name, mdl in models.items():
        pipe = Pipeline(steps=[('prep', preprocessor), ('model', mdl)])

        # Small GridSearch for RF, XGB, CatBoost
        if name == 'Random_Forest':
            param_grid = {
                'model__n_estimators': [300, 600],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 'log2']
            }
            gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
            gs.fit(X_tr, y_tr)
            pipe = gs.best_estimator_
        elif name == 'XGBRegressor' and XGBRegressor is not None:
            param_grid = {
                'model__n_estimators': [300, 600],
                'model__max_depth': [3, 5],
                'model__learning_rate': [0.05, 0.1],
                'model__subsample': [0.8, 1.0],
                'model__colsample_bytree': [0.8, 1.0]
            }
            gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
            gs.fit(X_tr, y_tr)
            pipe = gs.best_estimator_
        elif name == 'CatBoost' and CatBoostRegressor is not None:
            param_grid = {
                'model__depth': [4, 6, 8],
                'model__learning_rate': [0.03, 0.1],
                'model__iterations': [300, 600]
            }
            gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
            gs.fit(X_tr, y_tr)
            pipe = gs.best_estimator_
        else:
            pipe.fit(X_tr, y_tr)

        # Evaluate
        y_pred = pipe.predict(X_te)
        r2 = r2_score(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae = mean_absolute_error(y_te, y_pred)
        rows.append({'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae})
        trained[name] = pipe

        if args.cv:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_validate(pipe, X_tr, y_tr, scoring=("r2", "neg_root_mean_squared_error"), cv=cv, n_jobs=-1)
            mean_r2 = float(np.mean(cv_scores['test_r2']))
            mean_rmse = float(-np.mean(cv_scores['test_neg_root_mean_squared_error']))
            print(f"CV [{name}] R2: {mean_r2:.3f} | RMSE: {mean_rmse:.3f}")

    results_df = pd.DataFrame(rows).sort_values('R2', ascending=False)
    print('\nModel comparison (target=G3):')
    print(results_df)

    best_name = results_df.iloc[0]['Model']
    best_pipe = trained[best_name]
    print(f"\nBest model: {best_name}")

    # Try to extract feature importances
    feat_df = None
    try:
        num_names = num_cols
        try:
            cat_names = best_pipe.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_cols)
        except Exception:
            cat_names = best_pipe.named_steps['prep'].named_transformers_['cat'].get_feature_names(cat_cols)
        feature_names = list(num_names) + list(cat_names)

        model = best_pipe.named_steps['model']
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            coefs = np.ravel(model.coef_)
            importances = np.abs(coefs) / (np.abs(coefs).sum() + 1e-9)

        if importances is not None and len(importances) == len(feature_names):
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
            print('\nTop 20 features:')
            print(feat_df.head(20))
    except Exception as e:
        print('Feature importance extraction skipped:', e)

    os.makedirs('artifacts', exist_ok=True)
    results_df.to_csv('artifacts/model_comparison_g3.csv', index=False)
    if feat_df is not None:
        feat_df.to_csv('artifacts/feature_importance_g3.csv', index=False)

    # Save best model and preprocessor separately for PredictPipeline compatibility
    try:
        with open('artifacts/preprocessor.pkl', 'wb') as f:
            dill.dump(best_pipe.named_steps['prep'], f)
        with open('artifacts/model.pkl', 'wb') as f:
            dill.dump(best_pipe.named_steps['model'], f)
        print("\nSaved artifacts/preprocessor.pkl and artifacts/model.pkl")
    except Exception as e:
        print('Saving model/preprocessor failed:', e)

    # Save metrics for UI
    try:
        # Find the row for best_name
        row = next((r for r in rows if r['Model'] == best_name), None)
        metrics = {
            'model': best_name,
            'R2': row['R2'] if row else None,
            'RMSE': row['RMSE'] if row else None,
            'MAE': row['MAE'] if row else None,
        }
        with open('artifacts/metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("Saved artifacts/metrics.json")
    except Exception as e:
        print('Saving metrics failed:', e)


if __name__ == '__main__':
    main()
