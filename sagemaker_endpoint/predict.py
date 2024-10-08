import json
import xgboost as xgb

def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(f"{model_dir}/model.xgb")
    return model

def predict_fn(input_data, model):
    dmatrix = xgb.DMatrix(input_data)
    prediction = model.predict(dmatrix)
    
    # Assuming the model outputs probabilities for approval
    approval_probability = prediction[0]
    
    # Calculate interest rate and monthly payment based on the model's prediction
    base_rate = 5.0
    interest_rate = base_rate + (1 - approval_probability) * 5
    loan_amount = input_data['loan_amount']
    loan_term = input_data['loan_term']
    monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_term)
    
    return {
        'approval_probability': float(approval_probability),
        'interest_rate': float(interest_rate),
        'monthly_payment': float(monthly_payment)
    }

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")

def calculate_monthly_payment(loan_amount, interest_rate, loan_term):
    monthly_rate = interest_rate / 100 / 12
    payment = (loan_amount * monthly_rate * (1 + monthly_rate)**loan_term) / ((1 + monthly_rate)**loan_term - 1)
    return payment