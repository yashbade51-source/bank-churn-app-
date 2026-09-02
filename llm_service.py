"""
llm_service.py - AI Retention Copilot & Explanation Service
Provides AI-driven explanations and targeted customer retention strategies
using Ollama (local Llama 3.2), OpenAI API, or intelligent rule-based fallback.
"""

import os
import streamlit as st

def get_openai_api_key():
    """Retrieve OpenAI API key from Streamlit secrets or environment."""
    try:
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            key = st.secrets["OPENAI_API_KEY"]
            if key and not key.startswith("sk-proj-YOUR_KEY"):
                return key
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY", "")


def generate_rule_based_ai_explanation(customer_data, churn_probability, top_factors):
    """
    Intelligent fallback generator when no external LLM endpoint is reachable.
    Produces high-quality, structured banking retention insights.
    """
    factors_text = []
    if isinstance(top_factors, list):
        factors_text = [str(f) for f in top_factors]
    elif isinstance(top_factors, dict):
        factors_text = [f"{k}: {v}" for k, v in top_factors.items()]

    risk_level = "High" if churn_probability >= 50 else ("Medium" if churn_probability >= 30 else "Low")
    balance = float(customer_data.get("Balance", customer_data.get("Account Balance", 0)))
    num_products = int(customer_data.get("NumOfProducts", customer_data.get("Number of Products", 1)))
    is_active = customer_data.get("IsActiveMember", customer_data.get("Active Member", 1))
    is_active_bool = is_active in [1, "1", "Yes", True]
    satisfaction = int(customer_data.get("Satisfaction Score", 3))

    explanation = []
    explanation.append("### 🔍 FACTORS CONTRIBUTING TO CHURN:")
    explanation.append(f"The machine learning model identified a **{churn_probability:.1f}% churn risk** ({risk_level} Risk Level). Key behavioral drivers include:")
    
    if factors_text:
        for idx, factor in enumerate(factors_text[:4], 1):
            explanation.append(f"{idx}. **{factor}**: Significant driver impacting customer retention scoring.")
    else:
        explanation.append(f"1. **Account Activity**: {'Customer is active' if is_active_bool else 'Customer is currently inactive on primary digital/branch channels'}.")
        explanation.append(f"2. **Product Engagement**: Customer holds {num_products} product(s).")
        explanation.append(f"3. **Satisfaction Level**: Customer satisfaction rated at {satisfaction}/5.")

    explanation.append("\n### 💡 RECOMMENDED RETENTION ACTIONS:")
    if not is_active_bool or satisfaction <= 2:
        explanation.append("1. **Priority Relationship Outreach**: Schedule a proactive 1-on-1 consultation within 48 hours to resolve friction points.")
    else:
        explanation.append("1. **Relationship Review**: Conduct a tailored account review to align product benefits with customer financial goals.")

    if num_products <= 1:
        explanation.append("2. **Product Bundling & Incentives**: Offer fee-free premium checking or a high-yield savings add-on with cashback incentives.")
    elif balance > 80000:
        explanation.append("2. **Wealth & Tier Upgrades**: Invite the customer to our Preferred Tier with dedicated wealth advisory benefits.")
    else:
        explanation.append("2. **Engagement Campaign**: Enroll customer in targeted digital engagement programs and automated reward point bonuses.")

    explanation.append("3. **Loyalty Incentive**: Provide temporary rate discounts or bonus reward points upon maintaining account activity for 90 days.")

    return "\n".join(explanation)


def generate_churn_explanation(customer_data, churn_probability, top_factors, provider_preference="auto"):
    """
    Generate structured churn explanation and practical retention actions.
    
    Parameters:
    - customer_data (dict): Customer feature dictionary.
    - churn_probability (float): Churn probability in percent (0-100).
    - top_factors (list or dict): Significant risk / protective factors.
    - provider_preference (str): 'auto', 'ollama', 'openai', or 'fallback'.
    
    Returns:
    - (str, str): Tuple of (generated_text, provider_used)
    """
    prompt = f"""
You are an expert Senior Banking Retention and Customer Experience Strategist.
An XGBoost machine learning model has evaluated a bank customer:

- Predicted Churn Probability: {churn_probability:.2f}%
- Customer Profile:
{customer_data}

- Key Influencing Factors (SHAP / ML Analysis):
{top_factors}

Please provide your strategic response formatted in markdown with exactly two sections:

### 🔍 FACTORS CONTRIBUTING TO CHURN:
List and briefly explain the most critical behavioral and demographic factors increasing churn risk.

### 💡 SUGGESTIONS & RETENTION ACTIONS:
Provide 3 concrete, practical, and highly targeted actions the bank should immediately execute to retain this customer.

Keep the response concise, professional, and actionable. Do not hallucinate data not provided.
"""

    # 1. Try Ollama if preferred or auto
    if provider_preference in ["auto", "ollama"]:
        try:
            import ollama
            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}
            )
            content = response.get("message", {}).get("content", "")
            if content.strip():
                return content, "Ollama (Llama 3.2)"
        except Exception:
            if provider_preference == "ollama":
                pass

    # 2. Try OpenAI if preferred or auto
    openai_key = get_openai_api_key()
    if provider_preference in ["auto", "openai"] and openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional bank customer retention advisor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            content = completion.choices[0].message.content
            if content.strip():
                return content, "OpenAI (GPT-4o-mini)"
        except Exception:
            if provider_preference == "openai":
                pass

    # 3. Rule-based Expert AI Engine Fallback
    fallback_text = generate_rule_based_ai_explanation(customer_data, churn_probability, top_factors)
    return fallback_text, "Banking AI Rule Engine (Offline/Local)"
