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


def generate_churn_explanation(
    customer_data,
    churn_probability,
    risk_factors=None,
    protective_factors=None,
    predicted_churn=None,
    threshold_pct=50.0,
    model_name=None,
    # Back-compat: old callers may still pass a single merged list.
    top_factors=None,
):
    """
    Generate a structured churn explanation and retention actions using Ollama.

    Parameters:
    - customer_data (dict): Customer feature dictionary. Not sent to the LLM
      directly (see note below) -- kept in the signature for compatibility
      and potential future use (e.g. logging).
    - churn_probability (float): Churn probability in percent (0-100).
    - risk_factors (list): Factors that push this customer TOWARD churn.
    - protective_factors (list): Factors that push this customer TOWARD retention.
    - predicted_churn (bool): The model's actual predicted class at the deployment
      threshold. Used to choose an accurate section header instead of always
      framing the customer as a churn case.
    - threshold_pct (float): Deployment decision threshold, in percent, for context.
    - model_name (str): Ollama model name. Defaults to a cloud-available model
      when using Ollama Cloud, or "llama3.2" when running against a local daemon.
    - top_factors (list): Deprecated. If provided (and risk/protective are not),
      treated as risk factors only, for backward compatibility.

    Returns:
    - (str, str): tuple of (markdown explanation, provider label used --
      "Ollama Cloud" or "Local Ollama" -- so the caller can display it).
    """
    if risk_factors is None and top_factors is not None:
        risk_factors = top_factors
    risk_factors = risk_factors or []
    protective_factors = protective_factors or []

    if predicted_churn is None:
        predicted_churn = churn_probability >= threshold_pct

    # Only hand the LLM the specific, already-computed factors -- never the
    # full raw customer_data -- so it can't invent commentary on fields
    # (salary, card tier, etc.) that were never actually flagged as
    # significant by the model/SHAP step.
    factors_block = (
        f"Risk factors (push toward churn): {risk_factors if risk_factors else 'None identified'}\n"
        f"Protective factors (push toward retention): {protective_factors if protective_factors else 'None identified'}"
    )

    outcome_label = "CHURN RISK" if predicted_churn else "LIKELY TO STAY (RETAINED)"
    factors_header = (
        "### 🔍 FACTORS CONTRIBUTING TO CHURN:"
        if predicted_churn
        else "### 🔍 FACTORS SUPPORTING RETENTION (why this customer is low-risk):"
    )
    actions_header = (
        "### 💡 SUGGESTIONS & RETENTION ACTIONS:"
        if predicted_churn
        else "### 💡 SUGGESTIONS TO STRENGTHEN THE RELATIONSHIP:"
    )

    prompt = f"""
XGBoost predicted churn probability: {churn_probability:.2f}%
Deployment decision threshold: {threshold_pct:.0f}%
Model's actual prediction for this customer: {outcome_label}

{factors_block}

Give the response in exactly two sections:

{factors_header}
List ONLY the factors given above under "Risk factors" or "Protective factors"
(whichever section applies) and briefly explain each one. Do not discuss any
factor that was not explicitly listed above, and do not treat a protective
factor as if it contributes to churn.

{actions_header}
Give 3 practical actions the bank can take, appropriate to whether this
customer is actually predicted to churn or not.

Keep the response short and specific.
Do not make up factors that are not provided above.
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