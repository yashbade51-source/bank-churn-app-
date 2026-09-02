"""
llm_service.py - AI Retention Copilot powered by Ollama (Llama 3.2)
Provides AI-driven explanations and targeted customer retention strategies
using local Ollama.
"""

import ollama


def generate_churn_explanation(customer_data, churn_probability, top_factors, model_name="llama3.2"):
    """
    Generate structured churn explanation and practical retention actions using Ollama.
    
    Parameters:
    - customer_data (dict): Customer feature dictionary.
    - churn_probability (float): Churn probability in percent (0-100).
    - top_factors (list): Top risk and protective factors from ML/SHAP.
    - model_name (str): Ollama model name (default: "llama3.2").
    
    Returns:
    - str: Markdown explanation and actionable retention steps.
    """
    prompt = f"""
XGBoost predicted churn probability: {churn_probability:.2f}%

Customer data:
{customer_data}

Factors identified by the ML/SHAP analysis:
{top_factors}

Give the response in exactly two sections:

### 🔍 FACTORS CONTRIBUTING TO CHURN:
List the most important factors and briefly explain each one.

### 💡 SUGGESTIONS & RETENTION ACTIONS:
Give 3 practical actions the bank can take to retain this customer.

Keep the response short and specific.
Do not make up factors that are not provided.
"""

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
