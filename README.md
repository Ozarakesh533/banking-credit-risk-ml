# 💳 BarX Bank — Credit Risk Intelligence Dashboard

> A machine learning project that predicts customer loan defaults, segments customers by risk, and delivers actionable business recommendations — built with a premium dark banking dashboard UI.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Status](https://img.shields.io/badge/Status-Live-22c55e?style=for-the-badge)

---

## 🌐 Live Demo
👉 **[banking-credit-risk-ml.streamlit.app](https://banking-credit-risk-ml.streamlit.app)**

---

## 📌 What Is This Project?

Every bank faces one critical question before approving a loan:

> **"Will this customer repay the loan — or will they default?"**

Approving too many risky loans = financial losses.
Rejecting too many good customers = lost business.

This project builds a complete **ML-powered Credit Risk Intelligence System** that solves this end-to-end:

- 📊 Analyses **2,000 customer accounts** across 14 financial features
- 🤖 Trains and compares **3 ML models** with full evaluation metrics
- 🎯 Segments customers into **LOW / MEDIUM / HIGH** risk tiers
- 🔍 Performs **root cause analysis** of default drivers
- ⚡ Predicts default risk **in real-time** for any new customer
- 📋 Generates a complete **Executive Stakeholder Report**

---

## 🗂️ Project Structure

```
banking-credit-risk-ml/
│
├── app.py               ← Complete self-contained Streamlit dashboard
│                          (generates data + trains model + shows UI — all in one)
├── train_model.py       ← Standalone model training script (optional, local use)
├── requirements.txt     ← Python libraries needed
├── .gitignore           ← Excludes data/ and models/ folders from GitHub
└── README.md            ← This file
```

> ✅ **No setup needed beyond `pip install`** — `app.py` generates all synthetic data and trains the model automatically on first load using `@st.cache_data`. No CSV files or pre-trained models required.

---

## 🖥️ Dashboard — 5 Interactive Tabs

| Tab | What You See |
|---|---|
| 📊 **Risk Segments** | Customer risk distribution pie, default rate by segment, debt ratio & income breakdown |
| 🤖 **ML Model** | ROC-AUC curve with fill, predicted probability histogram, 3-model comparison table |
| 🔍 **Root Cause Analysis** | Feature importance bar chart, risk scatter plot, custom default driver table with badges |
| 🎯 **Live Prediction** | Sliders + inputs → instant default probability % + animated gauge meter |
| 📋 **Stakeholder Report** | Executive summary, key findings, 5 prioritised business recommendations |

---

## ▶️ How To Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Ozarakesh533/banking-credit-risk-ml.git
cd banking-credit-risk-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

Open `http://localhost:8501` — the app auto-generates data and trains the model on startup.

---

## 🏗️ How It Works — Step by Step

### 1️⃣ Synthetic Data Generation
2,000 realistic customer records created using NumPy with real-world financial patterns:

| Feature | Description | Example |
|---|---|---|
| `age` | Customer age | 35 yrs |
| `annual_income` | Yearly income (₹) | ₹4,20,000 |
| `debt_ratio` | Total debt ÷ income | 0.38 |
| `revolving_utilization` | Credit card usage vs limit | 0.52 |
| `missed_pmts_30_60` | Payments late 30–60 days | 1 |
| `missed_pmts_90plus` | Payments late 90+ days | 0 |
| `loan_amount` | Loan applied (₹) | ₹5,00,000 |
| `employment_years` | Years at current job | 6 |

**3 engineered features created:**
- `total_missed` = sum of all missed payment columns
- `risk_score` = weighted formula (debt ratio + utilization + missed payments)
- `income_to_loan` = income ÷ loan amount (affordability ratio)

---

### 2️⃣ Risk Segmentation (Business Rules)

```
risk_score < 0.20  →  🟢 LOW RISK    → Eligible for auto-approval
risk_score < 0.45  →  🟡 MEDIUM RISK → Manual review required  
risk_score ≥ 0.45  →  🔴 HIGH RISK   → Reject or add collateral
```

---

### 3️⃣ ML Model Training & Comparison

| Model | Description | AUC-ROC | F1 Score |
|---|---|---|---|
| Logistic Regression | Linear classifier — good baseline | 0.742 | 0.64 |
| **Random Forest ✅** | 150 decision trees — best balance | **0.763** | **0.69** |
| Gradient Boosting | Sequential error-correction trees | 0.761 | 0.68 |

Random Forest selected — best AUC, interpretable feature importance, handles class imbalance.

---

### 4️⃣ Live Prediction Widget

Enter customer details via sliders → model returns:
- Default probability % (0–100%)
- Risk classification: 🟢 LOW / 🟡 MEDIUM / 🔴 HIGH
- Animated gauge meter showing risk zone

---

## 🔍 Root Cause Analysis — Key Default Drivers

| Risk Driver | Default Rate | Recommended Action |
|---|---|---|
| Missed Payments (90+ days) | ~65% | Auto-route to NPA monitoring workflow |
| Revolving Utilization > 70% | ~48% | Send automated credit counselling alert |
| Debt Ratio > 60% | ~42% | Cap new credit exposure for segment |
| Annual Income < ₹20K | ~38% | Require co-applicant or collateral |
| Employment < 2 Years | ~35% | Reduce approved loan tenure |

---

## 💡 Business Impact (Simulated)

| Metric | Before ML System | After ML System |
|---|---|---|
| Review process | 100% manual | ~60% auto-approved |
| Time — low risk application | 3–7 days | < 2 hours |
| Default detection accuracy | ~50% (judgment) | 76.3% (AUC-ROC) |
| Officer workload | All applications | Medium + High risk only |
| NPA early warning | None | 7-day lead time |

---

## 🎨 Dashboard Design

Built with a **premium dark banking theme** — intentionally different from generic dashboards:

- **Dark background** (`#0a0e1a`) with layered card hierarchy
- **DM Sans + DM Mono** (Google Fonts) for professional typography
- **5 KPI cards** with colour-coded top borders per metric type
- **Transparent Plotly charts** that blend into the dark background
- **Custom HTML/CSS** for risk badges, RCA table, and prediction result card
- **Gauge meter** with risk zone colour bands (green / orange / red)
- **Styled tabs** replacing default Streamlit UI

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Pandas / NumPy** | Data generation and feature engineering |
| **Scikit-Learn** | Random Forest, Logistic Regression, Gradient Boosting, ROC metrics |
| **Plotly** | ROC curve, pie, histogram, box plot, scatter, gauge indicator |
| **Streamlit** | 5-tab dark-themed interactive web dashboard |
| **Google Fonts** | DM Sans + DM Mono premium typography |

---

## 💡 Skills Demonstrated

```
✅ Machine Learning — Random Forest, Logistic Regression, Gradient Boosting
✅ Feature Engineering — 14 features including 3 custom-engineered
✅ Model Evaluation — AUC-ROC, Precision, Recall, F1, ROC Curve
✅ Risk Segmentation — Business rule-based LOW / MEDIUM / HIGH tiers
✅ Root Cause Analysis — Feature importance + default driver breakdown
✅ Real-time Prediction — Live input widget with gauge visualisation
✅ Stakeholder Reporting — Executive summary with P0/P1/P2 recommendations
✅ Premium UI Design — Dark banking theme, custom CSS, DM Sans font
✅ Self-contained pipeline — No external data files or pre-trained models needed
```

---

## 🚀 Future Enhancements

- [ ] Add SHAP values for individual customer prediction explanation
- [ ] Connect to PostgreSQL database for real data ingestion
- [ ] Add XGBoost and LightGBM to model comparison
- [ ] Downloadable PDF version of the Stakeholder Report
- [ ] Model retraining trigger when AUC drops below threshold

---

## 👨‍💻 About

**Rakesh Oza** — Data Scientist & AI Engineer | Pune, India

[![Email](https://img.shields.io/badge/Email-ozarakesh533@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:ozarakesh533@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-rakeshoza-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/rakeshoza)
[![GitHub](https://img.shields.io/badge/GitHub-Ozarakesh533-181717?style=flat&logo=github&logoColor=white)](https://github.com/Ozarakesh533)

---

> ⚠️ **Disclaimer:** All data is 100% synthetic, generated using NumPy for learning and portfolio purposes only. Does not represent any real bank, customer, or financial institution.
