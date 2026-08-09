import os
import sqlite3
import pandas as pd

def run_queries(db_path="data/sba_loans.db", export_dir="data/powerbi_exports"):
    os.makedirs(export_dir, exist_ok=True)
    print(f"Running SQL analysis on SQLite DB at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Query 1: Industry Risk Ranking (using CTE and Window Function)
    industry_query = """
    WITH industry_stats AS (
        SELECT 
            naics_sector,
            COUNT(*) as total_loans,
            SUM(CASE WHEN status = 'default' THEN 1 ELSE 0 END) as defaulted_loans,
            SUM(CASE WHEN status IN ('default', 'paid-in-full') THEN 1 ELSE 0 END) as resolved_loans,
            AVG(loan_amount) as avg_loan_amount,
            SUM(loan_amount) as total_portfolio_amount
        FROM sba_loans
        GROUP BY naics_sector
    )
    SELECT 
        naics_sector,
        total_loans,
        defaulted_loans,
        resolved_loans,
        CASE WHEN resolved_loans > 0 
             THEN ROUND((defaulted_loans * 100.0 / resolved_loans), 2) 
             ELSE 0.0 
        END as default_rate_pct,
        ROUND(avg_loan_amount, 2) as avg_loan_amount,
        ROUND(total_portfolio_amount, 2) as total_portfolio_amount,
        DENSE_RANK() OVER (ORDER BY CASE WHEN resolved_loans > 0 THEN defaulted_loans * 100.0 / resolved_loans ELSE 0.0 END DESC) as risk_rank
    FROM industry_stats
    ORDER BY risk_rank ASC;
    """
    
    df_industry = pd.read_sql_query(industry_query, conn)
    industry_csv = os.path.join(export_dir, "industry_risk_ranking.csv")
    df_industry.to_csv(industry_csv, index=False)
    print(f"Exported Industry Risk Ranking to: {industry_csv}")
    
    # Query 2: State Risk Ranking (using CTE and Window Function)
    state_query = """
    WITH state_stats AS (
        SELECT 
            state,
            COUNT(*) as total_loans,
            SUM(CASE WHEN status = 'default' THEN 1 ELSE 0 END) as defaulted_loans,
            SUM(CASE WHEN status IN ('default', 'paid-in-full') THEN 1 ELSE 0 END) as resolved_loans,
            AVG(loan_amount) as avg_loan_amount,
            SUM(loan_amount) as total_portfolio_amount
        FROM sba_loans
        GROUP BY state
    )
    SELECT 
        state,
        total_loans,
        defaulted_loans,
        resolved_loans,
        CASE WHEN resolved_loans > 0 
             THEN ROUND((defaulted_loans * 100.0 / resolved_loans), 2) 
             ELSE 0.0 
        END as default_rate_pct,
        ROUND(avg_loan_amount, 2) as avg_loan_amount,
        ROUND(total_portfolio_amount, 2) as total_portfolio_amount,
        DENSE_RANK() OVER (ORDER BY CASE WHEN resolved_loans > 0 THEN defaulted_loans * 100.0 / resolved_loans ELSE 0.0 END DESC) as risk_rank
    FROM state_stats
    ORDER BY risk_rank ASC;
    """
    
    df_state = pd.read_sql_query(state_query, conn)
    state_csv = os.path.join(export_dir, "state_risk_ranking.csv")
    df_state.to_csv(state_csv, index=False)
    print(f"Exported State Risk Ranking to: {state_csv}")
    
    conn.close()
    print("SQL analysis completed successfully.")

if __name__ == '__main__':
    run_queries()
