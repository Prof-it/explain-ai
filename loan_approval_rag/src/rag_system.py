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

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)

        # Create embeddings - let sentence-transformers handle device placement
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create vector store
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)

        # Initialize local LLM using Phi-2
        model_name = "microsoft/phi-2"
        
        # Set up tokenizer with proper kwargs
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Ensure the pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model with proper configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Let accelerate handle device placement
            trust_remote_code=True
        )
        
        # Configure pipeline without explicit device
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
            return_full_text=False
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever()
        )

    def get_loan_decision(self, credit_score: float, annual_income: float, 
                         debt_ratio: float, years_employed: float) -> dict:
        """Get loan decision and explanation."""
        query = f"""
        Explain loan decision for an applicant with:
        - Credit Score: {credit_score}
        - Annual Income: ${annual_income}
        - Debt-to-Income Ratio: {debt_ratio*100}%
        - Years Employed: {years_employed}

        Based on the loan approval criteria, should this loan be approved? 
        Explain the decision in detail referencing specific criteria from the documents.
        """

        explanation = self.qa_chain.run(query)

        # Determine if approved based on hard criteria
        approved = (
            credit_score >= 650 and
            annual_income >= 30000 and
            debt_ratio <= 0.43 and
            years_employed >= 2
        )

        return {
            "approved": approved,
            "explanation": explanation,
            "criteria_met": {
                "credit_score": credit_score >= 650,
                "income": annual_income >= 30000,
                "debt_ratio": debt_ratio <= 0.43,
                "employment": years_employed >= 2
            }
        }