import os
import sqlite3
import pandas as pd
import numpy as np

def clean_loan_amount(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace('$', '').replace(',', '').strip()
    try:
        return float(s)
    except ValueError:
        return 0.0

def clean_naics(val):
    if pd.isna(val):
        return 'Unknown'
    s = str(val).strip().split('.')[0] # Remove decimals if float
    if len(s) < 2 or s == '0' or s == '00' or s.startswith('0'):
        return 'Unknown'
    return s[:2]

def clean_state(val):
    if pd.isna(val):
        return 'Unknown'
    s = str(val).strip().upper()
    if len(s) != 2:
        return 'Unknown'
    return s

def clean_date(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d-%b-%y', '%d-%b-%Y'):
        try:
            return pd.to_datetime(s, format=fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    try:
        return pd.to_datetime(s).strftime('%Y-%m-%d')
    except Exception:
        return None

def clean_status(val):
    if pd.isna(val):
        return 'current'
    s = str(val).strip().upper()
    if 'CHGOFF' in s or 'DEFAULT' in s:
        return 'default'
    elif 'P I F' in s or 'PIF' in s or 'PAID' in s:
        return 'paid-in-full'
    else:
        return 'current'

def run_ingestion(csv_path="data/FOIA_7a_FY2020_Present_asof_260630.csv", db_path="data/sba_loans.db", nrows=100000):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Real SBA loan dataset not found at expected path: {csv_path}")
        
    print(f"Reading real SBA data from: {csv_path} (Loading first {nrows} rows)...")
    # Read the first N rows
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"Loaded raw data shape: {df.shape}")
    
    # Apply cleaning and map to standard names
    print("Standardizing and cleaning columns...")
    df_clean = pd.DataFrame()
    
    # Map BorrName
    df_clean['borr_name'] = df['BorrName'].fillna('Unknown Borrower')
    
    # Map Program
    df_clean['program'] = df['Program'].fillna('7A')
    
    # Map GrossApproval -> loan_amount
    df_clean['loan_amount'] = df['GrossApproval'].apply(clean_loan_amount)
    
    # Map NaicsCode -> naics_sector
    df_clean['naics_sector'] = df['NaicsCode'].apply(clean_naics)
    
    # Map BorrState -> state
    df_clean['state'] = df['BorrState'].apply(clean_state)
    
    # Map TermInMonths -> term_months
    df_clean['term_months'] = pd.to_numeric(df['TermInMonths'], errors='coerce').fillna(60).astype(int)
    
    # Map ApprovalDate -> approval_date
    df_clean['approval_date'] = df['ApprovalDate'].apply(clean_date)
    
    # Map LoanStatus -> status
    df_clean['status'] = df['LoanStatus'].apply(clean_status)
    
    # Drop rows with null loan_amount or approval_date
    df_clean = df_clean.dropna(subset=['loan_amount', 'approval_date'])
    print(f"Cleaned data shape for database: {df_clean.shape}")
    
    # Write to SQLite
    print(f"Connecting to SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Write table
    df_clean.to_sql('sba_loans', conn, if_exists='replace', index=False)
    
    # Create indexes for optimal queries
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_state ON sba_loans(state);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_naics ON sba_loans(naics_sector);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON sba_loans(status);")
    conn.commit()
    
    conn.close()
    print("Database ingestion completed successfully.")

if __name__ == '__main__':
    run_ingestion()
