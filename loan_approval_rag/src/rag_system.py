from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os


class LoanApprovalRAG:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.setup_rag_pipeline()

    def setup_rag_pipeline(self):
        """Initialize the RAG pipeline components."""
        # Load documents
        loader = DirectoryLoader(self.docs_dir, glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Increased for more context
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)

        # Create embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create vector store
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)

        # Initialize local LLM using Phi-2
        model_name = "microsoft/phi-2"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure pipeline for balanced responses
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=768,  # Increased for complete responses
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
            do_sample=False
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
        )

    def get_loan_decision(self, credit_score: float, annual_income: float, 
                         debt_ratio: float, years_employed: float) -> dict:
        """Get loan decision and explanation."""
        # Check criteria
        criteria_met = {
            "credit_score": credit_score >= 650,
            "income": annual_income >= 30000,
            "debt_ratio": debt_ratio <= 0.43,
            "employment": years_employed >= 2
        }
        
        # Format criteria details
        criteria_details = {
            "credit_score": {
                "value": credit_score,
                "requirement": 650,
                "status": "meets" if criteria_met["credit_score"] else "below",
                "description": "Indicates risk level based on payment history"
            },
            "income": {
                "value": annual_income,
                "requirement": 30000,
                "status": "meets" if criteria_met["income"] else "below",
                "description": "Shows capacity for loan repayment"
            },
            "debt_ratio": {
                "value": debt_ratio * 100,
                "requirement": 43,
                "status": "meets" if criteria_met["debt_ratio"] else "exceeds",
                "description": "Measures current debt burden vs income"
            },
            "employment": {
                "value": years_employed,
                "requirement": 2,
                "status": "meets" if criteria_met["employment"] else "below",
                "description": "Shows income stability"
            }
        }
        
        # Generate focused prompt
        if all(criteria_met.values()):
            prompt = """
            Loan APPROVED. Explain each criterion's significance briefly:
            Credit Score: {credit_score}/650 - Payment reliability
            Income: ${income:,.0f}/$30,000 - Repayment capacity
            Debt Ratio: {debt_ratio:.1f}%/43% - Financial obligations
            Employment: {employment:.1f}/2 years - Income stability
            
            Keep each point to one clear sentence.
            """.format(
                credit_score=credit_score,
                income=annual_income,
                debt_ratio=debt_ratio * 100,
                employment=years_employed
            )
        else:
            unmet = [
                f"{name.replace('_', ' ').title()}: Current {details['value']:.1f}{' years' if name == 'employment' else '%' if name == 'debt_ratio' else ''} "
                f"(Need: {details['requirement']}{' years' if name == 'employment' else '%' if name == 'debt_ratio' else ''})"
                for name, details in criteria_details.items()
                if not criteria_met[name]
            ]
            
            prompt = f"""
            Loan DENIED. For each unmet criterion, provide:
            1. One sentence explaining its importance
            2. One specific improvement action
            3. A realistic target timeframe

            Unmet Criteria:
            {chr(10).join('- ' + c for c in unmet)}

            Keep explanations concise and direct. End with a brief improvement summary.
            Avoid phrases like 'As an AI' or 'Assistant:'.
            """

        # Get explanation from RAG system
        explanation = self.qa_chain.run(prompt).strip()
        
        # Remove any "Assistant:" prefix if present
        explanation = explanation.replace("Assistant:", "").strip()

        return {
            "approved": all(criteria_met.values()),
            "explanation": explanation,
            "criteria_met": criteria_met
        }