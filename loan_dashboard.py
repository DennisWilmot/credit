import streamlit as st

st.title("🎈 Example of Credit-risk App")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Function to generate synthetic data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    income = np.random.normal(50000, 15000, num_samples)
    credit_score = np.random.randint(300, 850, num_samples)
    employment_type = np.random.choice(['Full-time', 'Part-time', 'Self-employed'], num_samples)
    loan_amount = np.random.uniform(1000, 50000, num_samples)
    loan_duration = np.random.randint(6, 60, num_samples)
    debt_to_income = np.random.uniform(0.1, 0.5, num_samples)
    
    df = pd.DataFrame({
        'Income': income,
        'Credit_Score': credit_score,
        'Employment_Type': employment_type,
        'Loan_Amount': loan_amount,
        'Loan_Duration': loan_duration,
        'Debt_to_Income': debt_to_income
    })
    
    df['Interest_Rate'] = (15 - (df['Credit_Score'] / 850) * 5) + (df['Loan_Amount'] / 50000) * 2 + df['Debt_to_Income'] * 10
    
    df['Default'] = (df['Credit_Score'] < 600).astype(int)
    
    return df

st.title("My methodology")
project_description = """
Project Overview: Loan Risk Assessment and Interest Rate Optimization

This project focuses on building a loan risk assessment and interest rate optimization platform. The primary objective is to assess the risk associated with providing loans and dynamically adjust the interest rate to account for the likelihood of borrower default. The system leverages machine learning models, particularly logistic regression, to predict the probability of default (PD) based on borrower attributes such as income, credit score, loan amount, and debt-to-income ratio.

Using the predicted default probability (PD), along with assumptions on Loss Given Default (LGD) and Exposure at Default (EAD), the platform calculates the expected loss (EL) for each loan. A risk tolerance threshold is defined by the user, representing the maximum expected loss they are willing to accept. If the expected loss exceeds this threshold, the system increases the interest rate to mitigate the higher risk, ensuring that loans remain profitable.

Key steps include:
- Creating synthetic data for borrower attributes and loan terms.
- Data Input: Borrower attributes such as income, credit score, and loan amount are entered.
- Default Prediction: A logistic regression model predicts the likelihood of default.
- Expected Loss Calculation: The platform calculates the expected loss using the formula (EL = PD * LGD * EAD).
- Risk Tolerance: The user-defined risk tolerance helps determine whether the loan’s interest rate should be adjusted based on the expected loss.
- Interest Rate Optimization: If the expected loss exceeds the risk tolerance, the interest rate is increased to compensate for the elevated risk.

This platform enables lenders to make informed decisions, adjusting loan terms dynamically to ensure both profitability and risk management.
"""

synthetic_data_text = "created synthetic data for 1000 loan applicants with attributes such as income, credit score, employment type, loan amount, loan duration, and debt-to-income ratio. The interest rate was calculated based on the credit score, loan amount, and debt-to-income ratio, with additional assumptions for default probability and loss given default. Income was normally distributed with a mean of $50,000 and a standard deviation of $15,000. Credit scores ranged from 300 to 850, and employment types included full-time, part-time, and self-employed. Loan amounts varied from $1,000 to $50,000, with durations between 6 and 60 months. Debt-to-income ratios ranged from 0.1 to 0.5."


# Display project description in a read-only text area
st.text_area("Project Description", project_description, height=300)
# Streamlit UI
st.title("Explore Synthetic Loan Data")
st.text_area("Synthetic Data Generation", synthetic_data_text, height=200)


# Generate synthetic data
data = generate_synthetic_data()

# Display the data in the Streamlit app
st.subheader("Data Preview")
st.dataframe(data.head())  # Display the first few rows of the dataframe

# Provide a checkbox to show full data
if st.checkbox("Show full dataset"):
    st.write(data)

# Display some statistics about the dataset
st.subheader("Data Summary")
st.write(data.describe())

# Filter the data by employment type
employment_type_filter = st.selectbox("Select Employment Type", data['Employment_Type'].unique())
filtered_data = data[data['Employment_Type'] == employment_type_filter]

# Show filtered data
st.subheader(f"Data for Employment Type: {employment_type_filter}")
st.dataframe(filtered_data)

# Plot some basic statistics using matplotlib and Streamlit
st.subheader("Income Distribution")
fig, ax = plt.subplots()
ax.hist(data['Income'], bins=20)
st.pyplot(fig)

st.subheader("Credit Score Distribution")
fig, ax = plt.subplots()
ax.hist(data['Credit_Score'], bins=20)
st.pyplot(fig)

# Split the data
X = data[['Income', 'Credit_Score', 'Loan_Amount', 'Loan_Duration', 'Debt_to_Income']]
y = data['Interest_Rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression uses Default as the target, so we need to split it too
y_default = data['Default']
X_train_default, X_test_default, y_train_default, y_test_default = train_test_split(X, y_default, test_size=0.2, random_state=42)

# Initialize models
linear_model = LinearRegression()
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# Train the regression models
linear_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
elastic_net_model.fit(X_train, y_train)

# Train a logistic regression model for default prediction
logistic_model = LogisticRegression()
logistic_model.fit(X_train_default, y_train_default)

st.title('Loan Parameter Optimizer - SeedJA')


select_model_text = """I trained three different models to predict the interest rate for a loan based on borrower attributes. Select which model you'd like to use"""
default_text = """I also trained a logistic regression model to predict the probability of default based on the same borrower attributes. This helps us calculate the expected loss for each loan."""
st.text_area("Model Selection", select_model_text, height=100)

# Model selection
model_choice = st.radio(
    "Select a model:",
    ('Linear Regression', 'Random Forest', 'Elastic Net')
)

# User inputs
income = st.number_input('Income', min_value=1000, max_value=200000, value=50000)
credit_score = st.slider('Credit Score', min_value=300, max_value=850, value=650)
loan_amount = st.number_input('Loan Amount', min_value=1000, max_value=50000, value=10000)
loan_duration = st.slider('Loan Duration (months)', min_value=6, max_value=60, value=24)
debt_to_income = st.slider('Debt to Income Ratio', min_value=0.1, max_value=0.5, value=0.25)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Income': [income],
    'Credit_Score': [credit_score],
    'Loan_Amount': [loan_amount],
    'Loan_Duration': [loan_duration],
    'Debt_to_Income': [debt_to_income]
})

# Prediction based on the selected model
if model_choice == 'Linear Regression':
    predicted_interest_rate = linear_model.predict(input_data)[0]
elif model_choice == 'Random Forest':
    predicted_interest_rate = random_forest_model.predict(input_data)[0]
else:
    predicted_interest_rate = elastic_net_model.predict(input_data)[0]

# Calculate monthly payment
monthly_payment = loan_amount * (1 + predicted_interest_rate / 100) / loan_duration

# Logistic regression for default prediction
default_probability = logistic_model.predict_proba(input_data)[0][1]
st.subheader(f'Default Probability: {default_probability:.2f}')
st.text_area("Default Prediction", default_text, height=100)


# Calculate the expected loss (EL = PD * LGD * EAD)
LGD = 0.3  # Assume loss given default is 30% of the loan amount
EAD = loan_amount  # Exposure at default is the total loan amount
expected_loss = default_probability * LGD * EAD
st.subheader(f'Expected Loss: ${expected_loss:.2f}')
expected_loss_text = """The expected loss is calculated based on the default probability, loss given default, and exposure at default. This represents the estimated loss that the lender may incur if the borrower defaults on the loan."""
st.text_area("Expected Loss Calculation", expected_loss_text, height=100)

# Quantify risk tolerance
risk_threshold = st.slider('Risk Tolerance (Max Expected Loss)', 0.0, 5000.0, 1000.0)
if expected_loss > risk_threshold:
    predicted_interest_rate += 2  # Increase interest rate for higher risk
    st.warning(f"Loan refinanced due to high expected loss. New Interest Rate: {predicted_interest_rate:.2f}%")

risk_threshold_text = """The risk tolerance threshold represents the maximum expected loss that the lender is willing to accept for a loan. If the expected loss exceeds this threshold, the system increases the interest rate to mitigate the higher risk."""
st.text_area("Risk Tolerance", risk_threshold_text, height=100)

st.subheader(f'Optimal Interest Rate: {predicted_interest_rate:.2f}%')
optmia_interest_rate_text = """The optimal interest rate is calculated based on the borrower attributes and the risk tolerance threshold. If the expected loss exceeds the risk tolerance, the interest rate is increased to compensate for the elevated risk."""
st.text_area("Optimal Interest Rate", optmia_interest_rate_text, height=100)
st.subheader(f'Estimated Monthly Payment: ${monthly_payment:.2f}')

