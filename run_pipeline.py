import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingest import run_ingestion
from src.queries import run_queries
from src.model import train_risk_model
from src.scorecard import run_scorecard

def main():
    print("======================================================================")
    print("STARTING CREDIT RISK DECISION SUPPORT SYSTEM PIPELINE")
    print("======================================================================\n")
    
    # 1. Ingest Data
    print("[STEP 1/4] Running Data Ingestion & Database Setup...")
    run_ingestion()
    print("Step 1 Completed.\n")
    
    # 2. Run SQL Analytics
    print("[STEP 2/4] Executing SQL Analytics (CTEs and Window Functions)...")
    run_queries()
    print("Step 2 Completed.\n")
    
    # 3. Train Model and Score Loans
    print("[STEP 3/4] Training Risk Model & Scoring Database...")
    train_risk_model()
    print("Step 3 Completed.\n")
    
    # 4. Generate Scorecard & Expected Loss
    print("[STEP 4/4] Generating Strategic Scorecard & Expected Loss Summary...")
    run_scorecard()
    print("Step 4 Completed.\n")
    
    print("======================================================================")
    print("PIPELINE EXECUTED SUCCESSFULLY")
    print("======================================================================")

if __name__ == '__main__':
    main()
