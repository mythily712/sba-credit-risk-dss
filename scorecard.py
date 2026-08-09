import os
import sqlite3
import pandas as pd
import numpy as np

def run_scorecard(db_path="data/sba_loans.db", export_dir="data/powerbi_exports"):
    os.makedirs(export_dir, exist_ok=True)
    print(f"Generating Risk scorecard and Expected Loss analysis using database: {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Load all loans (including predicted_pd from model step)
    df = pd.read_sql_query("SELECT * FROM sba_loans", conn)
    
    # Check if predicted_pd exists
    if 'predicted_pd' not in df.columns:
        raise ValueError("predicted_pd not found in sba_loans table. Please run model.py first.")
        
    # 1. Calculate individual Expected Loss (EL = PD * Loan Amount)
    df['expected_loss'] = df['predicted_pd'] * df['loan_amount']
    
    # Save individual ELs back to the main database
    df.to_sql('sba_loans', conn, if_exists='replace', index=False)
    
    # 2. Segment-level analysis (naics_sector x state)
    # Group and aggregate
    segment_stats = df.groupby(['naics_sector', 'state']).agg(
        total_loans=('loan_amount', 'count'),
        total_portfolio_amount=('loan_amount', 'sum'),
        total_expected_loss=('expected_loss', 'sum'),
        defaulted_loans=('status', lambda x: (x == 'default').sum()),
        resolved_loans=('status', lambda x: x.isin(['default', 'paid-in-full']).sum())
    ).reset_index()
    
    # Calculate rates
    segment_stats['observed_default_rate_pct'] = np.where(
        segment_stats['resolved_loans'] > 0,
        (segment_stats['defaulted_loans'] * 100.0) / segment_stats['resolved_loans'],
        0.0
    )
    
    segment_stats['avg_loan_size'] = segment_stats['total_portfolio_amount'] / segment_stats['total_loans']
    segment_stats['expected_loss_ratio_pct'] = (segment_stats['total_expected_loss'] * 100.0) / segment_stats['total_portfolio_amount']
    
    # 3. Rule-based Recommended Actions
    # High-impact expected loss concentration threshold (e.g. top 10% expected loss segments)
    el_threshold = segment_stats['total_expected_loss'].quantile(0.90) if len(segment_stats) > 10 else 1e9
    
    def recommend_action(row):
        dr = row['observed_default_rate_pct']
        el = row['total_expected_loss']
        
        # Rule:
        # Tighten if default rate is high (>13%) OR if the absolute Expected Loss is highly concentrated (>= 90th percentile)
        if dr > 13.0 or el >= el_threshold:
            return 'Tighten'
        # Grow if default rate is low (<6%)
        elif dr < 6.0:
            return 'Grow'
        # Otherwise Maintain
        else:
            return 'Maintain'
            
    segment_stats['recommended_action'] = segment_stats.apply(recommend_action, axis=1)
    
    # Round metrics for reporting and Power BI ingestion
    segment_stats['observed_default_rate_pct'] = segment_stats['observed_default_rate_pct'].round(2)
    segment_stats['expected_loss_ratio_pct'] = segment_stats['expected_loss_ratio_pct'].round(2)
    segment_stats['total_portfolio_amount'] = segment_stats['total_portfolio_amount'].round(2)
    segment_stats['total_expected_loss'] = segment_stats['total_expected_loss'].round(2)
    segment_stats['avg_loan_size'] = segment_stats['avg_loan_size'].round(2)
    
    # Save segment scorecard to CSV
    scorecard_csv = os.path.join(export_dir, "segment_scorecard.csv")
    segment_stats.to_csv(scorecard_csv, index=False)
    print(f"Exported Segment Scorecard to: {scorecard_csv}")
    
    # Save segment scorecard to SQLite database
    segment_stats.to_sql('segment_scorecard', conn, if_exists='replace', index=False)
    
    # 4. Expected Loss Summary Table (Power BI export)
    # Overall summary by industry and state
    industry_el = df.groupby('naics_sector').agg(
        total_loans=('loan_amount', 'count'),
        total_portfolio_amount=('loan_amount', 'sum'),
        total_expected_loss=('expected_loss', 'sum')
    ).reset_index()
    industry_el['expected_loss_ratio_pct'] = ((industry_el['total_expected_loss'] / industry_el['total_portfolio_amount']) * 100).round(2)
    industry_el['total_portfolio_amount'] = industry_el['total_portfolio_amount'].round(2)
    industry_el['total_expected_loss'] = industry_el['total_expected_loss'].round(2)
    
    el_summary_csv = os.path.join(export_dir, "expected_loss_summary.csv")
    industry_el.to_csv(el_summary_csv, index=False)
    print(f"Exported Expected Loss Summary to: {el_summary_csv}")
    
    # Re-create indexes
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_state ON sba_loans(state);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_naics ON sba_loans(naics_sector);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON sba_loans(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_score_state ON segment_scorecard(state);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_score_naics ON segment_scorecard(naics_sector);")
    conn.commit()
    
    conn.close()
    print("Scorecard and Expected Loss analysis completed successfully.")

if __name__ == '__main__':
    run_scorecard()
