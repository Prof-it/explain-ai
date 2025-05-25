import streamlit as st
from document_loader import create_sample_documents
from rag_system import LoanApprovalSystem
import os
import sys

@st.cache_resource
def initialize_rag_system():
    try:
        # Get the absolute path to the loan_approval_rag directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        docs_dir = os.path.join(base_dir, "docs")
        create_sample_documents(docs_dir)
        return LoanApprovalSystem(docs_dir)
    except Exception as e:
        st.error(f"Error initializing the system: {str(e)}")
        st.stop()

def main():
    st.title("Loan Approval Assistant")
    st.write("Enter your information to get a loan decision explanation")

    # Initialize RAG system with caching
    rag_system = initialize_rag_system()

    with st.form("loan_application"):
        st.markdown("""
        ### Loan Application Form
        Please enter your information below. 
        """)
        
        credit_score = st.number_input(
            "Credit Score", 
            min_value=300,  # mean-3*std = 700-3*50
            max_value=850,  # Typical max credit score
            value=700,      # Mean from training
            help="Credit score typically ranges from 300-1000"
        )
        
        annual_income = st.number_input(
            "Annual Income ($)", 
            min_value=0,   # mean-2*std = 60k-2*20k
            max_value=120000,  # mean+3*std = 60k+3*20k
            value=60000,       # Mean from training
            step=1000,
            help="Annual income in dollars, at least 0"
        )
        
        debt_ratio = st.number_input(
            "Debt-to-Income Ratio (0-1)", 
            min_value=0.1,    # mean-2*std = 0.3-2*0.1
            max_value=2.0,    # mean+3*std = 0.3+3*0.1
            value=0.3,        # Mean from training
            format="%.2f",
            help="Debt-to-income ratio as a decimal (e.g., 0.3 = 30%). Max 2.0"
        )
        
        years_employed = st.number_input(
            "Years Employed", 
            min_value=-2.0,    # Can't be negative
            max_value=75.0,   # mean+3*std = 10+3*5
            value=10.0,       # Mean from training
            format="%.1f",
            help="Number of years in current employment, negative values for unemployed, must be larger than -2.0"
        )
        
        submitted = st.form_submit_button("Get Decision")
        
        if submitted:
            with st.spinner("Processing your application..."):
                # Get decision from RAG system
                result = rag_system.get_loan_decision(
                    credit_score, 
                    annual_income, 
                    debt_ratio, 
                    years_employed
                )
                
                # Display decision
                st.write("### Loan Decision")
                if result["approved"]:
                    st.success("✅ APPROVED")
                else:
                    st.error("❌ NOT APPROVED")
                
                # Display explanation
                st.write("### Detailed Explanation")
                st.write(result["explanation"])
                
                # Display criteria checklist based on actual decision rules
                st.write("### Decision Criteria")
                criteria = result["criteria_met"]
                
                # Helper function to format criteria status
                def format_criteria(name, status):
                    if status is None:
                        return f"⚫ {name} (not checked in decision path)"
                    elif status:
                        return f"✅ {name} meets requirements"
                    else:
                        return f"❌ {name} needs improvement"
                
                # Credit Score
                st.markdown(format_criteria(
                    "Credit Score",
                    criteria["credit_score"]
                ))
                
                # Income
                st.markdown(format_criteria(
                    "Income",
                    criteria["income"]
                ))
                
                # Debt Ratio
                st.markdown(format_criteria(
                    "Debt ratio",
                    criteria["debt_ratio"]
                ))
                
                # Employment
                st.markdown(format_criteria(
                    "Employment history",
                    criteria["employment"]
                ))

if __name__ == "__main__":
    main()