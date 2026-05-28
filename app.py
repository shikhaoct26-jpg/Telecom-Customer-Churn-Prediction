import gradio as gr
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open("churn_model.pkl", "rb"))

# Prediction function
def predict_churn(age, gender, tenure, usage, support_calls, delay, spend, last_int, subscription, contract):
    
    # Create the DataFrame with the EXACT names and order seen in the training CSV
    # Note: Using underscores as seen in your error logs (e.g., 'Gender_Male')
    data = pd.DataFrame({
        'Age': [age],
        'Tenure': [tenure],
        'Usage Frequency': [usage],
        'Support Calls': [support_calls],
        'Payment Delay': [delay],
        'Total Spend': [spend],
        'Last Interaction': [last_int],
        'Gender_Male': [1 if gender == "Male" else 0],
        'Subscription Type_Premium': [1 if subscription == "Premium" else 0],
        'Subscription Type_Standard': [1 if subscription == "Standard" else 0],
        'Contract Length_Monthly': [1 if contract == "Monthly" else 0],
        'Contract Length_Quarterly': [1 if contract == "Quarterly" else 0]
    })

    # Make prediction
    prediction = model.predict(data)[0]

    if prediction == 1:
        return "🔴 Customer Likely to Churn."
    else:
        return "🟢 Customer Will Stay."

# Define the Gradio Interface
inputs = [
    gr.Number(label="Age"),
    gr.Dropdown(["Male", "Female"], label="Gender"),
    gr.Slider(0, 60, label="Tenure (Months)"),
    gr.Slider(0, 30, label="Usage Frequency (Times/Month)"),
    gr.Number(label="Support Calls"),
    gr.Number(label="Payment Delay (Days)"),
    gr.Number(label="Total Spend ($)"),
    gr.Number(label="Days Since Last Interaction"),
    gr.Dropdown(["Basic", "Standard", "Premium"], label="Subscription Type"),
    gr.Dropdown(["Monthly", "Quarterly", "Annual"], label="Contract Length")
]

app = gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs="text",
    title="Customer Churn Predictor",
    description="Enter customer details to predict the likelihood of churn.",
     theme=gr.themes.Soft()
)

if __name__ == "__main__":
    app.launch()