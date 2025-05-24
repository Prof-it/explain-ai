import os

def create_sample_documents(docs_dir: str):
    """Create sample loan approval documents."""
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    documents = {
        "credit_score_rules.txt": """
Credit Score Requirements:
- Excellent (750+): Automatically qualifies for best rates
- Good (700-749): Generally approved with standard rates
- Fair (650-699): May require additional documentation
- Poor (below 650): High risk, typically declined
- Below 600: Automatic rejection

Explanation of Credit Score Impact:
High credit scores demonstrate reliable payment history and responsible credit management. 
Low scores indicate higher risk of default based on past credit behavior.
""",
        "income_requirements.txt": """
Income and Debt Requirements:
- Minimum annual income: $30,000
- Debt-to-Income ratio must be below 43%
- Monthly loan payment cannot exceed 28% of monthly income

Higher income levels provide greater assurance of repayment capability.
Debt-to-Income ratio shows ability to take on additional debt while maintaining financial stability.
""",
        "employment_history.txt": """
Employment History Guidelines:
- Minimum 2 years continuous employment preferred
- Current employment status must be verified
- Self-employed applicants need 2 years tax returns
- Job changes within same field are acceptable
- Recent unemployment may require explanation

Stable employment history indicates reliable income source and lower risk.
Frequent job changes or employment gaps may signal financial instability.
""",
        "approval_process.txt": """
Loan Approval Process:
1. Initial application review
2. Credit score check
3. Income verification
4. Employment history review
5. Debt ratio calculation
6. Final decision

All criteria must be met for approval. Failing any single criterion may result in rejection.
Borderline cases may be approved with additional conditions or higher rates.
"""
    }

    for filename, content in documents.items():
        with open(os.path.join(docs_dir, filename), 'w') as f:
            f.write(content)

    return docs_dir