# 🏦 Bank Customer Churn Prediction & Retention Platform

An integrated Machine Learning, Explainable AI (SHAP), and Generative AI Retention Copilot platform designed for banking relationship managers and analytics teams.

---

## 🌟 Key Features

1. **🔮 Machine Learning Churn Assessment**
   - High-performance **XGBoost Classifier** evaluating customer churn probability.
   - Comprehensive customer profile input (demographics, balance, activity, products, loyalty attributes).
   - Dynamic Risk Level categorization (`LOW`, `MEDIUM`, `HIGH`, `VERY HIGH`).

2. **🔍 Explainable AI (SHAP Explainer)**
   - Real-time `shap.TreeExplainer` computing local feature impacts.
   - Distinct breakdown of **Risk Drivers** (factors elevating churn risk) and **Protective Factors** (attributes supporting retention).

3. **🤖 Generative AI Retention Copilot**
   - Natural language retention insights and tactical action plans.
   - Multi-provider support:
     - 🦙 **Ollama (Llama 3.2)** for local offline inference.
     - ✨ **OpenAI (GPT-4o-mini)** via `.streamlit/secrets.toml` or `OPENAI_API_KEY`.
     - 📋 **Banking Expert Rule Engine** fallback when offline.

4. **📊 Interactive Portfolio Analytics (EDA Dashboard)**
   - Portfolio overview with KPI tiles (Total Customers, Retained, Churned, Churn Rate).
   - Dynamic filtering by Geography and Active Membership.
   - Visualizations: Churn Distribution, Churn by Geography, Active Membership, Product Counts, and Age Cohorts.

5. **📁 Batch Customer Data Upload**
   - CSV uploader for batch customer evaluation.
   - Automatic schema validation and summary metrics calculation.

---

## 📁 Project Structure

```text
├── app.py                      # Main Streamlit web application
├── llm_service.py              # AI Copilot service (Ollama, OpenAI, Rule Engine)
├── test_model.py               # Comprehensive verification test suite
├── best_churn_model.pkl        # Trained XGBoost classification model
├── feature_columns.pkl         # Expected model feature schema
├── churn_config.pkl            # Model configuration and decision threshold
├── Customer-Churn-Records.csv  # Bank customer records dataset
├── Hackathon_project.ipynb     # Model training & EDA Jupyter Notebook
├── requirements.txt            # Python dependencies
└── .streamlit/
    ├── config.toml             # Streamlit visual theme configuration
    └── secrets.toml            # Optional API keys (OpenAI, etc.)
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Test Suite
```bash
python test_model.py
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚙️ AI Copilot Configuration (Optional)

- **Local Ollama**: Ensure Ollama is running (`ollama run llama3.2`).
- **OpenAI**: Add your key in `.streamlit/secrets.toml`:
  ```toml
  OPENAI_API_KEY = "your-api-key-here"
  ```
- **Fallback Engine**: If no LLM endpoint is reachable, the built-in banking rule engine automatically provides structured, context-aware retention strategies without interruption.
