"""
llm_service.py - AI Retention Copilot powered by Ollama
Provides AI-driven explanations and targeted customer retention strategies
using Ollama. Works both locally (local daemon on localhost:11434) and on
hosted platforms (Ollama Cloud API), depending on whether OLLAMA_API_KEY
is configured.
"""

import os
import ollama


def _get_api_key():
    """Look for OLLAMA_API_KEY in env vars, then in Streamlit secrets
    (Streamlit Cloud stores secrets separately from the environment)."""
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("OLLAMA_API_KEY")
        except Exception:
            pass
    return api_key


def _get_client():
    """
    Build an Ollama client.

    - If an API key is found, route requests to Ollama's hosted Cloud API
      (https://ollama.com) so no local daemon is required.
    - Otherwise, fall back to the local Ollama daemon (localhost:11434),
      which is what you want when running/testing on your own machine.
    """
    api_key = _get_api_key()

    if api_key:
        return ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"},
        )

    return ollama.Client()  # defaults to local http://localhost:11434


# Cloud-hosted models only include the larger models, not llama3.2.
# When OLLAMA_API_KEY is set we default to a cloud-available model;
# locally you can keep using llama3.2 if you have it pulled.
DEFAULT_LOCAL_MODEL = "llama3.2"
DEFAULT_CLOUD_MODEL = "gpt-oss:20b"


def generate_churn_explanation(customer_data, churn_probability, top_factors, model_name=None):
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

    client = _get_client()
    using_cloud = _get_api_key() is not None

    if model_name is None:
        model_name = DEFAULT_CLOUD_MODEL if using_cloud else DEFAULT_LOCAL_MODEL

    provider_used = "Ollama Cloud" if using_cloud else "Local Ollama"

    try:
        response = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    except ConnectionError as e:
        if using_cloud:
            raise ConnectionError(
                "Could not reach Ollama Cloud (https://ollama.com). "
                "Check your internet connection and that OLLAMA_API_KEY is valid."
            ) from e
        else:
            raise ConnectionError(
                "No OLLAMA_API_KEY found, so this tried to reach a local Ollama "
                "server at http://localhost:11434 and failed. Either set "
                "OLLAMA_API_KEY (in .streamlit/secrets.toml or as an env var) to "
                "use Ollama Cloud, or install and start Ollama locally "
                "(ollama run llama3.2)."
            ) from e

    return response["message"]["content"], provider_used