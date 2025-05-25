from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.tree import DecisionTreeClassifier
import torch
import os
import re
import joblib
import numpy as np


class LoanApprovalSystem:
    def __init__(self, docs_dir: str):
        """Initialize the loan approval system."""
        self.docs_dir = docs_dir
        self.setup_explanation_model()
        self.load_decision_tree()
        
        # Feature names and normalization parameters from training data
        self.feature_config = {
            'income': {'mean': 60000, 'std': 20000},
            'age': {'mean': 40, 'std': 10},
            'years_employed': {'mean': 10, 'std': 5},
            'debt_ratio': {'mean': 0.3, 'std': 0.1},
            'credit_score': {'mean': 700, 'std': 50}
        }
        self.feature_names = list(self.feature_config.keys())
        
        # Fixed approval message
        self.approval_message = "Congratulations! Your loan application has been approved. Please contact our loan department to proceed with the next steps."

    def load_decision_tree(self):
        """Load the pre-trained decision tree model."""
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model", "decision_tree.pkl")
        self.model = joblib.load(model_path)
        
        # Print tree structure for debugging
        tree = self.model.tree_
        print("\nDecision Tree Structure:")
        print(f"Number of nodes: {tree.node_count}")
        print(f"Features at nodes: {tree.feature}")
        print(f"Thresholds: {tree.threshold}")
        print(f"Values at nodes: {tree.value}")
        print(f"Children left: {tree.children_left}")
        print(f"Children right: {tree.children_right}")

    def normalize_text(self, text: str) -> str:
        """Normalize text to ensure proper formatting."""
        # Fix common formatting issues
        text = text.replace('−', '-')  # Replace special minus with hyphen
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        
        # Fix number formatting
        def fix_number(match):
            return match.group(0).replace(',', '')
        text = re.sub(r'\d{1,3}(?:,\d{3})+', fix_number, text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])([^\s\d])', r'\1 \2', text)
        
        # Ensure proper spacing around numbers
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
        
        return text

    def clean_output(self, text: str, is_denial: bool) -> str:
        """Clean and format the model output."""
        # Basic cleaning
        text = text.strip()
        
        # Capitalize first letter if needed
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Add final period if missing
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
            
        return text

    # def setup_explanation_model(self):
    #     """Initialize the explanation model - using BLOOMZ-1b7 for better reasoning with smaller footprint."""
    #     model_name = "bigscience/bloomz-1b7"
        
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         device_map="auto",
    #         torch_dtype=torch.float32
    #     )
        
    #     # Configure pipeline for text generation
    #     self.pipeline = pipeline(
    #         task="text-generation",
    #         model=model,
    #         tokenizer=tokenizer,
    #         max_new_tokens=200,     # Control new token generation
    #         temperature=0.7,        # Moderate randomness
    #         top_k=50,              # Limit vocabulary choices
    #         num_beams=3,           # Beam search for coherent output
    #         early_stopping=True,    # Stop when complete
    #         num_return_sequences=1,
    #         pad_token_id=tokenizer.eos_token_id,
    #         eos_token_id=tokenizer.eos_token_id
    #     )
    def setup_explanation_model(self):
        model_name = "microsoft/phi-2"  # or "microsoft/phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float32
        )
        self.pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.7,
            top_k=50,
            num_beams=3,
            early_stopping=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    def normalize_features(self, features: dict) -> np.ndarray:
        """Normalize features using training data parameters."""
        normalized = []
        for name in self.feature_names:
            value = features[name]
            config = self.feature_config[name]
            normalized_value = (value - config['mean']) / config['std']
            normalized.append(normalized_value)
        return np.array(normalized)

    def get_decision_path(self, features: np.ndarray, raw_features: dict) -> list:
        """Extract the decision path from the tree for the given features."""
        # Get the decision path
        node_indicator = self.model.decision_path([features])
        leaf_id = self.model.apply([features])
        
        print("\nDetailed Decision Path Debug:")
        print(f"Node indicator shape: {node_indicator.shape}")
        print(f"Node indicator indices: {node_indicator.indices}")
        print(f"Node indicator indptr: {node_indicator.indptr}")
        print(f"Leaf ID: {leaf_id}")
        
        # Get the path
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        print(f"Node index sequence: {node_index}")
        
        # Extract decision rules
        rules = []
        print("\nDecision Tree Path Trace:")  # Debug output
        for node_id in node_index:
            if node_id != leaf_id[0]:  # Not a leaf node
                feature_idx = self.model.tree_.feature[node_id]
                if feature_idx >= 0:  # Not a leaf
                    threshold = self.model.tree_.threshold[node_id]
                    feature_name = self.feature_names[feature_idx]
                    actual_value = features[feature_idx]
                    
                    # Get raw (unnormalized) values for display
                    config = self.feature_config[feature_name]
                    raw_threshold = threshold * config['std'] + config['mean']
                    raw_value = raw_features[feature_name]
                    
                    direction = "left" if actual_value <= threshold else "right"
                    
                    # Debug output for each node
                    print(f"\nNode {node_id} debug:")
                    print(f"Feature index: {feature_idx}")
                    print(f"Feature name: {feature_name}")
                    print(f"Normalized value: {actual_value:.2f}")
                    print(f"Normalized threshold: {threshold:.2f}")
                    print(f"Raw value: {raw_value:.2f}")
                    print(f"Raw threshold: {raw_threshold:.2f}")
                    print(f"Direction: {direction}")
                    print(f"Next node: {self.model.tree_.children_left[node_id] if direction == 'left' else self.model.tree_.children_right[node_id]}")
                    
                    rules.append({
                        'node_id': node_id,
                        'feature': feature_name,
                        'operation': '<=' if direction == 'left' else '>',
                        'threshold': raw_threshold,
                        'actual': raw_value,
                        'direction': direction
                    })
                    print(f"Node {node_id}: {feature_name} = {raw_value:.2f} {rules[-1]['operation']} {raw_threshold:.2f} (went {direction})")
        return rules

    def generate_explanation(self, decision_path: list, features: dict, is_approved: bool) -> str:
        if is_approved:
            return "Congratulations! Your loan application has been approved. Please contact our loan department to proceed with the next steps."
        if not decision_path:
            return "We regret to inform you that your loan application cannot be approved at this time."

        # Format feature values and names consistently
        def format_value(feature: str, value: float) -> str:
            if feature == 'credit_score':
                return f"{value:.0f}"
            elif feature == 'debt_ratio':
                return f"{value*100:.1f}%"
            elif feature == 'years_employed':
                return f"{value:.1f} years"
            elif feature == 'income':
                return f"${value:,.0f}"
            return str(value)

        def format_feature(feature: str) -> str:
            if feature == 'years_employed':
                return 'employment history'
            elif feature == 'debt_ratio':
                return 'debt-to-income ratio'
            return feature.replace('_', ' ')

        # Build path description for LLM
        path_info = []
        for i, rule in enumerate(decision_path):
            feature = rule['feature']
            value = rule['actual']
            operation = rule['operation']
            is_last = i == len(decision_path) - 1
            
            formatted_value = format_value(feature, value)
            formatted_feature = format_feature(feature)
            
            if not is_last:
                path_info.append(f"PASSED: {formatted_feature} of {formatted_value}")
            else:  # Rejection reason
                if operation == '>':
                    path_info.append(f"FAILED: {formatted_feature} of {formatted_value} (needs to be lower)")
                elif operation == '<=':
                    path_info.append(f"FAILED: {formatted_feature} of {formatted_value} (needs to be higher)")


        # Only explain the last failed node
        # fail_info = []
        # if decision_path:
        #     last_rule = decision_path[-1]
        #     feature = last_rule['feature']
        #     value = last_rule['actual']
        #     operation = last_rule['operation']
        #     formatted_value = format_value(feature, value)
        #     formatted_feature = format_feature(feature)
        #     if operation == '>':
        #         fail_info = f"{formatted_feature} of {formatted_value} is too low."
        #     elif operation == '<=':
        #         fail_info = f"{formatted_feature} of {formatted_value} is too high."
        #     else:
        #         fail_info = f"{formatted_feature} of {formatted_value} did not meet the requirement."
        # else:
        #     fail_info = "Unknown reason."

        # Build a readable path string for the LLM
        path_str = "\n".join(path_info)

        last_failed = path_info[-1].replace("FAILED: ", "")
        # Remove the value, keep only the feature and direction
        if "credit score" in last_failed and "higher" in last_failed:
            reason = "credit score is too low"
        elif "income" in last_failed and "higher" in last_failed:
            reason = "income is too low"
        elif "debt-to-income ratio" in last_failed and "lower" in last_failed:
            reason = "debt-to-income ratio is too high"
        elif "employment history" in last_failed and "higher" in last_failed:
            reason = "employment history is too short"
        else:
            reason = last_failed

        prompt = (
            f"Write a polite, single-sentence explanation for a loan rejection due to {reason}. "
            "Do not mention any minimum, maximum, or required value. "
            "Instead, suggest how the applicant could improve this aspect for future applications."
        )

        try:
            assert self.pipeline is not None and self.pipeline.tokenizer is not None, "Pipeline or tokenizer not loaded"
            result = self.pipeline(
                prompt,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                eos_token_id=self.pipeline.tokenizer.eos_token_id
            )
            if result is None:
                raise RuntimeError("Pipeline returned None. Check pipeline initialization and input.")
            outputs = list(result)
            print("DEBUG: outputs[0] =", outputs[0], "type:", type(outputs[0]))

            if isinstance(outputs[0], dict):
                response = outputs[0].get('generated_text', str(outputs[0]))
            else:
                response = str(outputs[0])

            explanation = response.strip()

            # Remove prompt echo if present
            if explanation.startswith(prompt):
                explanation = explanation[len(prompt):].strip()

            # Extract only the part after 'Output:'
            if "Output:" in explanation:
                explanation = explanation.split("Output:", 1)[1].strip()
            elif "output:" in explanation:
                explanation = explanation.split("output:", 1)[1].strip()

            print("\nLLM Debug Info:")
            print("Prompt:", prompt)
            print("Raw Response:", response)
            print("Final Explanation:", explanation)

            return explanation

        except Exception as e:
            print(f"Error generating LLM explanation: {str(e)}")
            # Fallback explanation if LLM fails
            if len(path_info) > 1:
                passed_parts = [p.replace("PASSED: ", "") for p in path_info[:-1]]
                failed_part = path_info[-1].replace("FAILED: ", "").split(" (needs")[0]
                return f"We regret to inform you that your loan application cannot be approved at this time. While you have acceptable {', '.join(passed_parts)}, your {failed_part}."
            else:
                failed_part = path_info[0].replace("FAILED: ", "").split(" (needs")[0]
                return f"We regret to inform you that your loan application cannot be approved at this time because your {failed_part}."

    def get_loan_decision(self, credit_score: float, annual_income: float, 
                         debt_ratio: float, years_employed: float) -> dict:
        """Get loan decision and explanation using the decision tree model."""
        # Prepare raw features dictionary
        raw_features = {
            'income': annual_income,
            'age': 40,  # Mean from training data
            'years_employed': years_employed,
            'debt_ratio': debt_ratio,
            'credit_score': credit_score
        }
        
        print("\nRaw Input Features:")  # Debug output
        for name, value in raw_features.items():
            print(f"- {name}: {value}")
        
        # Normalize features
        normalized_features = self.normalize_features(raw_features)
        print("\nNormalized Features:")  # Debug output
        for name, value in zip(self.feature_names, normalized_features):
            print(f"- {name}: {value:.2f}")
        
        # Get model prediction
        raw_prediction = self.model.predict([normalized_features])[0]
        probabilities = self.model.predict_proba([normalized_features])[0]
        print(f"\nRaw prediction value: {raw_prediction}")
        print(f"Prediction probabilities: {probabilities}")  # [prob_reject, prob_approve]
        is_approved = bool(raw_prediction)
        print(f"\nDecision Tree Prediction: {'APPROVED' if is_approved else 'REJECTED'}")
        
        # Get decision path and generate explanation
        decision_path = self.get_decision_path(normalized_features, raw_features)
        
        # Initialize all criteria as None (not checked)
        criteria_met = {
            "credit_score": None,
            "income": None,
            "debt_ratio": None,
            "employment": None
        }
        
        # Process decision path
        if is_approved:
            # If approved, all checked criteria are met
            for rule in decision_path:
                feature = rule['feature']
                criteria_key = 'employment' if feature == 'years_employed' else feature
                if criteria_key in criteria_met:
                    criteria_met[criteria_key] = True
        else:
            # If rejected, all nodes except the last one were passed
            for i, rule in enumerate(decision_path):
                feature = rule['feature']
                criteria_key = 'employment' if feature == 'years_employed' else feature
                if criteria_key in criteria_met:
                    # Last node in path is the failing condition
                    criteria_met[criteria_key] = (i < len(decision_path) - 1)
        
        # Generate explanation based on the last node if rejected
        explanation = self.generate_explanation(
            [decision_path[-1]] if decision_path and not is_approved else [], 
            raw_features, 
            is_approved
        )

        return {
            "approved": is_approved,
            "explanation": explanation,
            "criteria_met": criteria_met
        }