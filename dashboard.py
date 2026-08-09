import os
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as objects
import streamlit as st
import json

# Set page config
st.set_page_config(
    page_title="SBA Credit Risk Decision Support System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #007bff;
        margin-bottom: 20px;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #212529;
    }
    .recommendation-grow {
        background-color: #d4edda;
        color: #155724;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .recommendation-maintain {
        background-color: #fff3cd;
        color: #856404;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .recommendation-tighten {
        background-color: #f8d7da;
        color: #721c24;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Database Helper
DB_PATH = "data/sba_loans.db"
MODEL_PARAMS_PATH = "data/powerbi_exports/model_params.json"

@st.cache_data
def load_data():
    if not os.path.exists(DB_PATH):
        return None, None
    conn = sqlite3.connect(DB_PATH)
    df_loans = pd.read_sql_query("SELECT * FROM sba_loans", conn)
    df_scorecard = pd.read_sql_query("SELECT * FROM segment_scorecard", conn)
    conn.close()
    
    # Parse dates
    df_loans['approval_date'] = pd.to_datetime(df_loans['approval_date'])
    df_loans['year'] = df_loans['approval_date'].dt.year
    
    return df_loans, df_scorecard

# NAICS mapping
NAICS_MAP = {
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'Wholesale Trade',
    '44': 'Retail Trade',
    '45': 'Retail Trade',
    '54': 'Professional Services',
    '62': 'Healthcare',
    '72': 'Accommodation & Food Services',
    '81': 'Other Services',
    'Unknown': 'Unknown Sectors'
}

# Main Application
def main():
    st.title("🏦 SBA Credit Risk Decision Support System")
    st.subheader("Portfolio Growth Optimization & Underwriting Risk Analytics")
    
    df_loans, df_scorecard = load_data()
    
    if df_loans is None:
        st.error("⚠️ Database not found! Please run the data pipeline orchestrator (`run_pipeline.py`) first to generate the dataset and train the model.")
        return
        
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/120/university.png", width=80)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Dashboard View:",
        ["📊 Executive Summary", "🗺️ Risk Heatmaps & Trends", "📋 Segment Scorecard", "🔮 Underwriting Simulator"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Portfolio At A Glance")
    total_port = df_loans['loan_amount'].sum()
    st.sidebar.metric("Total Portfolio Value", f"${total_port:,.2f}")
    
    resolved = df_loans[df_loans['status'].isin(['default', 'paid-in-full'])]
    def_rate = (resolved['status'] == 'default').mean() * 100
    st.sidebar.metric("Average Default Rate", f"{def_rate:.2f}%")
    
    # ------------------ Page 1: Executive Summary ------------------
    if page == "📊 Executive Summary":
        st.header("Executive Summary & Strategic Insights")
        
        # Calculate key metrics
        total_loans = len(df_loans)
        avg_loan = df_loans['loan_amount'].mean()
        total_el = df_loans['expected_loss'].sum()
        el_ratio = (total_el / total_port) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #28a745;">
                <div class="metric-label">Active Portfolio Size</div>
                <div class="metric-value">${total_port/1e6:.1f}M</div>
                <div style="font-size: 12px; color: #6c757d;">Total Loans: {total_loans:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #dc3545;">
                <div class="metric-label">Avg. Default Rate (DR)</div>
                <div class="metric-value">{def_rate:.2f}%</div>
                <div style="font-size: 12px; color: #6c757d;">Based on Resolved Loans</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #ffc107;">
                <div class="metric-label">Total Expected Loss (EL)</div>
                <div class="metric-value">${total_el/1e6:.2f}M</div>
                <div style="font-size: 12px; color: #6c757d;">EL Ratio: {el_ratio:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #17a2b8;">
                <div class="metric-label">Average Loan Size</div>
                <div class="metric-value">${avg_loan/1e3:.1f}K</div>
                <div style="font-size: 12px; color: #6c757d;">Weighted across all terms</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.subheader("💡 Automated Credit Policy Recommendations")
        
        # Calculate insights dynamically
        # 1. Best industries to grow (DR < 6%)
        grow_inds = df_scorecard[df_scorecard['recommended_action'] == 'Grow'].groupby('naics_sector')['total_loans'].sum().reset_index()
        grow_inds['industry_name'] = grow_inds['naics_sector'].map(NAICS_MAP)
        grow_list = grow_inds.sort_values(by='total_loans', ascending=False)['industry_name'].head(3).tolist()
        
        # 2. Worst industries to tighten (DR > 13%)
        tight_inds = df_scorecard[df_scorecard['recommended_action'] == 'Tighten'].groupby('naics_sector')['total_expected_loss'].sum().reset_index()
        tight_inds['industry_name'] = tight_inds['naics_sector'].map(NAICS_MAP)
        tight_list = tight_inds.sort_values(by='total_expected_loss', ascending=False)['industry_name'].head(3).tolist()
        
        # 3. Top Expected Loss states
        top_el_states = df_loans.groupby('state')['expected_loss'].sum().reset_index()
        top_el_states = top_el_states.sort_values(by='expected_loss', ascending=False).head(3)['state'].tolist()
        
        # Recommendation Cards
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.markdown("### 📈 Strategic Expansion Opportunities (Grow)")
            st.markdown(f"""
            * **Target Sectors**: **{', '.join(grow_list) if grow_list else 'Healthcare, Professional Services'}** are demonstrating extremely strong credit performance. 
              - *Action*: Run targeted marketing campaigns and streamline approvals for loans in these industries.
            * **Low-Risk Geographies**: Geographies with low defaults and low EL ratios should be prioritized for capital deployment.
            * **Underwriting Adjustment**: For these sectors, consider increasing maximum leverage ratios and relaxing secondary collateral requirements.
            """)
            
        with rec_col2:
            st.markdown("### ⚠️ Underwriting Restructuring Areas (Tighten)")
            st.markdown(f"""
            * **Risk Exposure Sectors**: **{', '.join(tight_list) if tight_list else 'Accommodation & Food Services, Retail Trade'}** are contributing disproportionately to default counts and Expected Loss.
              - *Action*: Tighten credit policy, require higher debt-service coverage ratios (DSCR), and demand stronger guarantor backing.
            * **High-Concentration Geographies**: States like **{', '.join(top_el_states)}** show elevated Expected Loss volumes.
              - *Action*: Reduce single-state credit concentration limits.
            * **Policy Rule**: Implement mandatory manual reviews for all loans exceeding $500,000 in these risk segments.
            """)
            
        st.markdown("---")
        st.subheader("📊 Portfolio Exposure Breakdown")
        
        # Industry exposure chart (Portfolio Amount vs Expected Loss)
        ind_summary = df_loans.groupby('naics_sector').agg(
            portfolio_amt=('loan_amount', 'sum'),
            expected_loss=('expected_loss', 'sum')
        ).reset_index()
        ind_summary['industry'] = ind_summary['naics_sector'].map(NAICS_MAP)
        
        fig = px.bar(
            ind_summary, 
            x='industry', 
            y=['portfolio_amt', 'expected_loss'],
            title="Portfolio Exposure and Expected Loss by Industry",
            labels={'value': 'USD ($)', 'industry': 'NAICS Sector'},
            barmode='group',
            color_discrete_sequence=['#007bff', '#dc3545']
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ Page 2: Risk Heatmaps & Trends ------------------
    elif page == "🗺️ Risk Heatmaps & Trends":
        st.header("Risk Heatmaps & Historical Lending Trends")
        
        # Interactive heatmap filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            metric_choice = st.selectbox(
                "Select Heatmap Metric:",
                ["Observed Default Rate (%)", "Total Expected Loss ($)"]
            )
            
        # Pivot table for Heatmap
        if metric_choice == "Observed Default Rate (%)":
            pivot_df = df_scorecard.pivot(index='state', columns='naics_sector', values='observed_default_rate_pct').fillna(0)
            title = "Default Rate Heatmap by State and NAICS Industry"
            color_scale = "Reds"
        else:
            pivot_df = df_scorecard.pivot(index='state', columns='naics_sector', values='total_expected_loss').fillna(0)
            title = "Total Expected Loss Heatmap by State and NAICS Industry"
            color_scale = "Oranges"
            
        # Map columns to Names for better tooltips
        pivot_df.columns = [f"{col} - {NAICS_MAP.get(col, '')}" for col in pivot_df.columns]
        
        fig_heat = px.imshow(
            pivot_df,
            labels=dict(x="Industry Sector", y="Borrower State", color=metric_choice),
            title=title,
            color_continuous_scale=color_scale,
            aspect="auto"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📈 Historical Lending & Performance Trends")
        
        # Group loans by Year for trends
        yearly_stats = df_loans.groupby('year').agg(
            total_loans=('loan_amount', 'count'),
            total_portfolio=('loan_amount', 'sum'),
            defaulted_loans=('status', lambda x: (x == 'default').sum()),
            resolved_loans=('status', lambda x: x.isin(['default', 'paid-in-full']).sum())
        ).reset_index()
        
        yearly_stats['default_rate'] = np.where(
            yearly_stats['resolved_loans'] > 0,
            (yearly_stats['defaulted_loans'] / yearly_stats['resolved_loans']) * 100,
            0.0
        )
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_vol = px.line(
                yearly_stats,
                x='year',
                y='total_portfolio',
                title="Lending Volume (Approved Dollars) Over Time",
                labels={'total_portfolio': 'Total Approved Loans ($)', 'year': 'Year'},
                markers=True
            )
            st.plotly_chart(fig_vol, use_container_width=True)
            
        with col_t2:
            fig_def = px.line(
                yearly_stats,
                x='year',
                y='default_rate',
                title="Default Rate Trend Over Time (Resolved Portfolios)",
                labels={'default_rate': 'Default Rate (%)', 'year': 'Year'},
                markers=True,
                line_shape="linear",
                color_discrete_sequence=['#dc3545']
            )
            st.plotly_chart(fig_def, use_container_width=True)

    # ------------------ Page 3: Segment Scorecard ------------------
    elif page == "📋 Segment Scorecard":
        st.header("Risk-vs-Growth Segment Scorecard")
        st.markdown("Filter and inspect granular segments (State x NAICS 2-digit) to inform portfolio allocation and underwriting standards.")
        
        # Filters
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            states_list = ['All'] + sorted(df_scorecard['state'].unique().tolist())
            selected_state = st.selectbox("Filter State:", states_list)
        with f_col2:
            naics_list = ['All'] + sorted(df_scorecard['naics_sector'].unique().tolist())
            selected_naics = st.selectbox("Filter NAICS Sector:", naics_list)
        with f_col3:
            action_list = ['All'] + sorted(df_scorecard['recommended_action'].unique().tolist())
            selected_action = st.selectbox("Filter Recommended Action:", action_list)
            
        # Apply filters
        filtered_df = df_scorecard.copy()
        if selected_state != 'All':
            filtered_df = filtered_df[filtered_df['state'] == selected_state]
        if selected_naics != 'All':
            filtered_df = filtered_df[filtered_df['naics_sector'] == selected_naics]
        if selected_action != 'All':
            filtered_df = filtered_df[filtered_df['recommended_action'] == selected_action]
            
        # Add Industry Descriptions
        filtered_df['industry_name'] = filtered_df['naics_sector'].map(NAICS_MAP)
        
        # Reorder and rename columns for display
        display_df = filtered_df[[
            'state', 'naics_sector', 'industry_name', 'total_loans', 
            'total_portfolio_amount', 'total_expected_loss', 
            'observed_default_rate_pct', 'expected_loss_ratio_pct', 'recommended_action'
        ]].copy()
        
        display_df.columns = [
            'State', 'NAICS Sector', 'Industry Name', 'Loan Count', 
            'Portfolio Size ($)', 'Expected Loss ($)', 'Default Rate (%)', 
            'Expected Loss Ratio (%)', 'Recommended Action'
        ]
        
        # Style table function
        def style_action(val):
            if val == 'Grow':
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif val == 'Maintain':
                return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
            elif val == 'Tighten':
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            return ''
            
        st.dataframe(
            display_df.style.applymap(style_action, subset=['Recommended Action'])
            .format({
                'Portfolio Size ($)': '${:,.2f}',
                'Expected Loss ($)': '${:,.2f}',
                'Default Rate (%)': '{:.2f}%',
                'Expected Loss Ratio (%)': '{:.2f}%'
            }),
            use_container_width=True,
            height=500
        )

    # ------------------ Page 4: Underwriting Simulator ------------------
    elif page == "🔮 Underwriting Simulator":
        st.header("🔮 Loan Underwriting Simulator")
        st.markdown("Input loan application parameters below to predict default probability and expected loss, and get automated underwriting decisions based on our credit model.")
        
        if not os.path.exists(MODEL_PARAMS_PATH):
            st.error("⚠️ Model parameter file not found! Please run the pipeline script to train the model first.")
            return
            
        # Form
        sim_col1, sim_col2 = st.columns([1, 1.2])
        
        with sim_col1:
            st.subheader("📋 Loan Application Details")
            
            with st.form("underwrite_form"):
                loan_amt = st.number_input("Requested Loan Amount ($):", min_value=5000, max_value=5000000, value=250000, step=10000)
                term = st.slider("Loan Term (Months):", min_value=12, max_value=360, value=84, step=12)
                
                # Fetch available states and NAICS sectors
                states_sim = sorted(df_loans['state'].unique().tolist())
                state = st.selectbox("Borrower State:", states_sim)
                
                naics_sim = sorted([c for c in df_loans['naics_sector'].unique().tolist() if c != 'Unknown'])
                naics = st.selectbox(
                    "NAICS Industry Sector:", 
                    naics_sim, 
                    format_func=lambda x: f"{x} - {NAICS_MAP.get(x, '')}"
                )
                
                submit_button = st.form_submit_button("Run Risk Assessment")
                
        # Handle predictions
        if submit_button or 'submitted' in st.session_state:
            st.session_state['submitted'] = True
            
            from src.model import score_single_loan
            
            # Predict default probability (PD)
            pd_val = score_single_loan(loan_amt, term, naics, state, MODEL_PARAMS_PATH)
            el_val = pd_val * loan_amt
            
            # Action recommendation logic (consistent with scorecard rule)
            dr_pct = pd_val * 100
            
            # Fetch segment average details
            segment_match = df_scorecard[(df_scorecard['state'] == state) & (df_scorecard['naics_sector'] == naics)]
            if len(segment_match) > 0:
                seg_default_rate = segment_match.iloc[0]['observed_default_rate_pct']
                seg_action = segment_match.iloc[0]['recommended_action']
            else:
                seg_default_rate = dr_pct
                seg_action = 'Maintain'
                
            # Decision mapping
            if dr_pct > 13.0 or seg_action == 'Tighten':
                action_class = "recommendation-tighten"
                action_text = "TIGHTEN (DECLINE / ENHANCED VETTING)"
                desc_color = "red"
            elif dr_pct < 6.0 and seg_action == 'Grow':
                action_class = "recommendation-grow"
                action_text = "GROW (PRE-APPROVE / EXPEDITE)"
                desc_color = "green"
            else:
                action_class = "recommendation-maintain"
                action_text = "MAINTAIN (STANDARD VETTING)"
                desc_color = "orange"
                
            with sim_col2:
                st.subheader("🔍 Credit Risk Assessment Output")
                
                # Output HTML Card
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-radius: 8px; padding: 25px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08); margin-bottom: 20px;">
                    <div style="font-size: 14px; color: #6c757d; font-weight: bold; text-transform: uppercase;">Underwriting Policy Recommendation</div>
                    <div style="margin-top: 5px; margin-bottom: 20px;">
                        <span class="{action_class}" style="font-size: 20px; padding: 8px 15px;">{action_text}</span>
                    </div>
                    <hr style="margin: 15px 0;">
                    <div class="row" style="display: flex; justify-content: space-between;">
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 12px; color: #6c757d; font-weight: bold;">PROBABILITY OF DEFAULT</div>
                            <div style="font-size: 32px; font-weight: bold; color: {desc_color};">{pd_val*100:.2f}%</div>
                        </div>
                        <div style="text-align: center; flex: 1; border-left: 1px solid #dee2e6;">
                            <div style="font-size: 12px; color: #6c757d; font-weight: bold;">EXPECTED LOSS</div>
                            <div style="font-size: 32px; font-weight: bold; color: {desc_color};">${el_val:,.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Business explanation section
                st.markdown("### 💬 Plain-English Risk Explanation")
                
                # Load coefficients for driver analysis
                with open("data/powerbi_exports/model_params.json", "r") as f:
                    params = json.load(f)
                coef_dict = params['coef_dict']
                
                # Retrieve individual factors
                naics_coef = coef_dict.get(f"naics_sector_{naics}", 0.0)
                state_coef = coef_dict.get(f"state_{state}", 0.0)
                
                # Evaluate factors
                reasons = []
                
                # Industry Impact
                ind_name = NAICS_MAP.get(naics, "")
                if naics_coef > 0.3:
                    reasons.append(f"🔴 **High-Risk Industry**: The loan is in **{ind_name}** (NAICS {naics}), which historically exhibits elevated rates of default in the portfolio. This adds significant risk to the application.")
                elif naics_coef < -0.3:
                    reasons.append(f"🟢 **Low-Risk Industry**: The loan is in **{ind_name}** (NAICS {naics}), which is a stable, low-default industry sector.")
                else:
                    reasons.append(f"⚪ **Moderate Industry Risk**: Industry **{ind_name}** has standard/average risk levels.")
                    
                # Term Impact
                if term <= 60:
                    reasons.append(f"🔴 **Short Term**: A loan term of **{term} months** increases default risk, typically associated with tight cash-flow amortizations or higher-risk working capital requirements.")
                elif term >= 240:
                    reasons.append(f"🟢 **Long Term (Asset-Backed)**: A long term of **{term} months** is indicative of real estate or heavy equipment backing, reducing default likelihood significantly.")
                    
                # Loan Amount Impact
                if loan_amt > 1000000:
                    reasons.append(f"🟢 **Large Loan Vetting**: While the dollar exposure (${loan_amt:,.2f}) is high, larger loans generally require rigorous collateralization and financial covenants, which historically leads to lower default probabilities.")
                elif loan_amt < 100000:
                    reasons.append(f"🔴 **Small Business Cashflow**: Smaller loan sizes (< $100,000) are typically uncollateralized or highly sensitive to short-term working capital shortages, pushing risk up.")
                    
                # Geography Impact
                if state_coef > 0.15:
                    reasons.append(f"🔴 **State Risk Multiplier**: State **{state}** exhibits a higher default rate multiplier in historical SBA reviews, slightly elevating risk.")
                elif state_coef < -0.15:
                    reasons.append(f"🟢 **Stable State Economy**: State **{state}** shows a strong economic repayment record, decreasing overall credit risk.")
                    
                # Output drivers
                for r in reasons:
                    st.write(r)

if __name__ == '__main__':
    main()
