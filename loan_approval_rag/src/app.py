import streamlit as st
from document_loader import create_sample_documents
from rag_system import LoanApprovalRAG
import os
import sys

@st.cache_resource
def initialize_rag_system():
    try:
        docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
        create_sample_documents(docs_dir)
        return LoanApprovalRAG(docs_dir)
    except Exception as e:
        st.error(f"Error initializing the system: {str(e)}")
        st.stop()

def main():
    st.title("Loan Approval Assistant")
    st.write("Enter your information to get a loan decision explanation")

    # Initialize RAG system with caching
    rag_system = initialize_rag_system()

    with st.form("loan_application"):
        credit_score = st.number_input(
            "Credit Score", 
            min_value=300, 
            max_value=850,
            value=700
        )
        
        annual_income = st.number_input(
            "Annual Income ($)", 
            min_value=0, 
            max_value=1000000,
            value=50000
        )
        
        debt_ratio = st.number_input(
            "Debt-to-Income Ratio", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3,
            format="%.2f"
        )
        
        years_employed = st.number_input(
            "Years Employed", 
            min_value=0.0, 
            max_value=50.0, 
            value=3.0,
            format="%.1f"
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
                
                # Display criteria checklist
                st.write("### Criteria Checklist")
                criteria = result["criteria_met"]
                
                if criteria["credit_score"]:
                    st.success("✅ Credit Score meets requirements")
                else:
                    st.error("❌ Credit Score below minimum (650)")
                    
                if criteria["income"]:
                    st.success("✅ Income meets requirements")
                else:
                    st.error("❌ Income below minimum ($30,000)")
                    
                if criteria["debt_ratio"]:
                    st.success("✅ Debt ratio acceptable")
                else:
                    st.error("❌ Debt ratio too high (max 43%)")
                    
                if criteria["employment"]:
                    st.success("✅ Employment history sufficient")
                else:
                    st.error("❌ Employment history insufficient (min 2 years)")

if __name__ == "__main__":
    main()