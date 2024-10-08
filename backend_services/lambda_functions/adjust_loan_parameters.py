import json
import boto3
import os
from aws_lambda_powertools import Metrics, Logger

sagemaker_runtime = boto3.client('sagemaker-runtime')
bedrock_runtime = boto3.client('bedrock-runtime')

MODEL_ENDPOINT = os.environ['MODEL_ENDPOINT']

metrics = Metrics()
logger = Logger()

@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        adjusted_loan = adjust_loan_parameters(body)
        
        return {
            'statusCode': 200,
            'body': json.dumps(adjusted_loan)
        }
    except Exception as e:
        logger.exception("Error adjusting loan parameters")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def adjust_loan_parameters(loan_data):
    # Prepare data for model prediction
    model_input = prepare_model_input(loan_data)
    
    # Get model prediction
    prediction = get_model_prediction(model_input)
    
    # Process prediction and generate adjusted loan
    adjusted_loan = process_prediction(prediction, loan_data)
    
    # Generate explanation
    explanation = generate_explanation(prediction, model_input)
    adjusted_loan['explanation'] = explanation
    
    return adjusted_loan

def prepare_model_input(loan_data):
    # Ensure all required fields are present
    required_fields = ['loan_amount', 'loan_term', 'credit_score', 'income', 'debt_to_income_ratio']
    for field in required_fields:
        if field not in loan_data:
            raise ValueError(f"Missing required field: {field}")
    
    return json.dumps(loan_data)

def get_model_prediction(model_input):
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=MODEL_ENDPOINT,
        ContentType='application/json',
        Body=model_input
    )
    return json.loads(response['Body'].read().decode())

def process_prediction(prediction, loan_data):
    # Process the prediction and generate adjusted loan details
    # This is a simplified example; you'd need to implement the actual logic
    adjusted_loan = {
        'loan_amount': loan_data['loan_amount'],
        'loan_term': loan_data['loan_term'],
        'interest_rate': prediction['interest_rate'],
        'monthly_payment': prediction['monthly_payment'],
        'approval_probability': prediction['approval_probability']
    }
    return adjusted_loan

def generate_explanation(prediction, features):
    prompt = f"""
    Given the following loan application details:
    {json.dumps(features, indent=2)}
    
    And the model's prediction:
    {json.dumps(prediction, indent=2)}
    
    Provide a brief, easy-to-understand explanation of how each factor influenced the loan terms and approval probability.
    """
    
    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-v2',
        body=json.dumps({
            'prompt': prompt,
            'max_tokens_to_sample': 300,
            'temperature': 0.7,
            'top_p': 0.95,
        })
    )
    
    explanation = json.loads(response['body'])['completion']
    return explanation

# Add this Lambda function to your API Gateway