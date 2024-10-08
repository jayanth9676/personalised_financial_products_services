import json
import boto3
import os
from aws_lambda_powertools import Metrics, Logger
from aws_lambda_powertools.metrics import MetricUnit

sagemaker_runtime = boto3.client('sagemaker-runtime')

MODEL_ENDPOINT = os.environ['MODEL_ENDPOINT']

metrics = Metrics()
logger = Logger()

@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        updated_offer = update_loan_offer(body)
        
        return {
            'statusCode': 200,
            'body': json.dumps(updated_offer)
        }
    except Exception as e:
        logger.exception("Error updating loan offer")
        metrics.add_metric(name="UpdateErrors", unit=MetricUnit.Count, value=1)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def update_loan_offer(offer_data):
    # Prepare data for model prediction
    model_input = prepare_model_input(offer_data)
    
    # Get model prediction
    prediction = get_model_prediction(model_input)
    
    # Process prediction and generate updated offer
    updated_offer = process_prediction(prediction, offer_data)
    
    return updated_offer

def prepare_model_input(offer_data):
    # Transform offer data into model input format
    model_input = {
        'loan_amount': offer_data['loan_amount'],
        'loan_term': offer_data['loan_term'],
        'credit_score': offer_data['credit_score'],
        'income': offer_data['income'],
        'debt_to_income_ratio': offer_data['debt_to_income_ratio']
    }
    return json.dumps(model_input)

def get_model_prediction(model_input):
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=MODEL_ENDPOINT,
        ContentType='application/json',
        Body=model_input
    )
    return json.loads(response['Body'].read().decode())

def process_prediction(prediction, offer_data):
    approval_probability = prediction[0]
    
    updated_offer = {
        'loan_amount': offer_data['loan_amount'],
        'loan_term': offer_data['loan_term'],
        'approval_probability': approval_probability,
        'interest_rate': calculate_interest_rate(approval_probability, offer_data),
        'monthly_payment': calculate_monthly_payment(offer_data['loan_amount'], offer_data['loan_term'], approval_probability)
    }
    
    return updated_offer

def calculate_interest_rate(approval_probability, offer_data):
    base_rate = 5.0
    risk_adjustment = (1 - approval_probability) * 10
    credit_score_adjustment = (800 - offer_data['credit_score']) / 100
    return round(base_rate + risk_adjustment + credit_score_adjustment, 2)

def calculate_monthly_payment(loan_amount, loan_term, approval_probability):
    interest_rate = calculate_interest_rate(approval_probability, {'credit_score': 700})  # Assuming average credit score
    monthly_rate = interest_rate / 100 / 12
    num_payments = loan_term
    
    monthly_payment = (loan_amount * monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    
    return round(monthly_payment, 2)