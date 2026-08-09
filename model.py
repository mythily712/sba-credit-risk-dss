import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json

def train_risk_model(db_path="data/sba_loans.db", export_dir="data/powerbi_exports"):
    os.makedirs(export_dir, exist_ok=True)
    print(f"Training Credit Risk model using database: {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Load all loans
    df_all = pd.read_sql_query("SELECT * FROM sba_loans", conn)
    print(f"Total loans loaded: {len(df_all)}")
    
    # Training data: only resolved loans (default = 1, paid-in-full = 0)
    df_train_val = df_all[df_all['status'].isin(['default', 'paid-in-full'])].copy()
    df_train_val['target'] = (df_train_val['status'] == 'default').astype(int)
    print(f"Resolved loans for training/evaluation: {len(df_train_val)} (default rate: {df_train_val['target'].mean()*100:.2f}%)")
    
    # Features
    cat_features = ['naics_sector', 'state']
    num_features = ['loan_amount', 'term_months']
    
    # One-hot encoding and scaling
    # We fit on df_all to make sure we handle all categories if scoring
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df_all[cat_features])
    
    scaler = StandardScaler()
    scaler.fit(df_all[num_features])
    
    # Transform training data
    X_cat = encoder.transform(df_train_val[cat_features])
    X_num = scaler.transform(df_train_val[num_features])
    X = np.hstack([X_num, X_cat])
    y = df_train_val['target'].values
    
    # Column names for interpretation
    cat_cols = encoder.get_feature_names_out(cat_features)
    feature_names = num_features + list(cat_cols)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_prob))
    }
    print(f"Model Metrics: {metrics}")
    
    # Save metrics
    with open(os.path.join(export_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Feature coefficients (importances)
    coefs = model.coef_[0]
    df_coefs = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'abs_coefficient': np.abs(coefs)
    }).sort_values(by='abs_coefficient', ascending=False)
    
    coefs_csv = os.path.join(export_dir, "model_feature_importance.csv")
    df_coefs.to_csv(coefs_csv, index=False)
    print(f"Exported Feature Importance to: {coefs_csv}")
    
    # Business interpretation dictionary
    intercept = float(model.intercept_[0])
    
    # Separate numeric, industry, and state coefficients
    interpret_dict = {
        'intercept': intercept,
        'numerical': {},
        'industry_impact': {},
        'state_impact': {}
    }
    
    # Numeric features scaling info
    interpret_dict['numerical_scaling'] = {
        'loan_amount': {'mean': float(scaler.mean_[0]), 'scale': float(scaler.scale_[0])},
        'term_months': {'mean': float(scaler.mean_[1]), 'scale': float(scaler.scale_[1])}
    }
    
    for row in df_coefs.itertuples():
        feat = row.feature
        coef = float(row.coefficient)
        if feat in num_features:
            interpret_dict['numerical'][feat] = coef
        elif feat.startswith('naics_sector_'):
            sector = feat.replace('naics_sector_', '')
            interpret_dict['industry_impact'][sector] = coef
        elif feat.startswith('state_'):
            state = feat.replace('state_', '')
            interpret_dict['state_impact'][state] = coef
            
    # Save model parameters for in-memory scoring in dashboard/pipeline
    model_params = {
        'intercept': intercept,
        'coef_dict': dict(zip(feature_names, coefs)),
        'cat_cols': list(cat_cols),
        'categories': [list(cat) for cat in encoder.categories_],
        'num_features': num_features,
        'scaler_mean': list(scaler.mean_),
        'scaler_scale': list(scaler.scale_)
    }
    with open(os.path.join(export_dir, "model_params.json"), "w") as f:
        json.dump(model_params, f, indent=4)
        
    # Scoring the entire database (all loans)
    print("Scoring the entire loan database...")
    X_all_cat = encoder.transform(df_all[cat_features])
    X_all_num = scaler.transform(df_all[num_features])
    X_all = np.hstack([X_all_num, X_all_cat])
    
    df_all['predicted_pd'] = model.predict_proba(X_all)[:, 1]
    
    # Write back to SQLite database
    df_all.to_sql('sba_loans', conn, if_exists='replace', index=False)
    
    # Re-create indexes
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_state ON sba_loans(state);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_naics ON sba_loans(naics_sector);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON sba_loans(status);")
    conn.commit()
    
    conn.close()
    print("Model training and scoring completed successfully.")
    
def score_single_loan(loan_amount, term_months, naics_sector, state, params_path="data/powerbi_exports/model_params.json"):
    """Scoring helper for dashboard or single predictions without loading full model objects."""
    with open(params_path, 'r') as f:
        params = json.load(f)
        
    coef_dict = params['coef_dict']
    intercept = params['intercept']
    
    # Scale numerical values
    mean_amt, scale_amt = params['scaler_mean'][0], params['scaler_scale'][0]
    mean_term, scale_term = params['scaler_mean'][1], params['scaler_scale'][1]
    
    scaled_amt = (loan_amount - mean_amt) / scale_amt
    scaled_term = (term_months - mean_term) / scale_term
    
    # Sum up model linear combination
    z = intercept
    z += scaled_amt * coef_dict.get('loan_amount', 0.0)
    z += scaled_term * coef_dict.get('term_months', 0.0)
    
    # Add category hot-encoded values
    naics_key = f"naics_sector_{naics_sector}"
    state_key = f"state_{state}"
    
    z += coef_dict.get(naics_key, 0.0)
    z += coef_dict.get(state_key, 0.0)
    
    # Apply sigmoid
    pd_val = 1.0 / (1.0 + np.exp(-z))
    return pd_val

if __name__ == '__main__':
    train_risk_model()
