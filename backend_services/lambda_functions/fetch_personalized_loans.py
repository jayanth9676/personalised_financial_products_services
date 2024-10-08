import json
import boto3
import os
from aws_lambda_powertools import Metrics, Logger
import shap

sagemaker_runtime = boto3.client('sagemaker-runtime')
bedrock_runtime = boto3.client('bedrock-runtime')

MODEL_ENDPOINT = os.environ['MODEL_ENDPOINT']
RAG_INDEX = os.environ['RAG_INDEX']

metrics = Metrics()
logger = Logger()

@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event, context):
    try:
        user_id = event['requestContext']['authorizer']['claims']['sub']
        user_data = get_user_data(user_id)
        
        personalized_loans = get_personalized_loans(user_data)
        
        return {
            'statusCode': 200,
            'body': json.dumps(personalized_loans)
        }
    except Exception as e:
        logger.exception("Error fetching personalized loans")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def get_user_data(user_id):
    # Implement actual DynamoDB query to fetch user data
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(os.environ['USERS_TABLE_NAME'])
    response = table.get_item(Key={'user_id': user_id})
    return response.get('Item', {})

def get_personalized_loans(user_data):
    model_input = prepare_model_input(user_data)
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=MODEL_ENDPOINT,
        ContentType='application/json',
        Body=json.dumps(model_input)
    )
    
    model_output = json.loads(response['Body'].read().decode())
    
    personalized_loans = generate_loan_options(model_output, user_data)
    
    # Generate SHAP values for explainability
    explainer = shap.TreeExplainer(model)  # Assuming 'model' is available
    shap_values = explainer.shap_values(model_input)
    
    # Add explanations to personalized loans
    for i, loan in enumerate(personalized_loans):
        loan['explanation'] = generate_explanation(shap_values[i], model_input)
    
    return personalized_loans

def generate_explanation(shap_values, features):
    explanation = []
    for feature, value in zip(features.keys(), shap_values):
        impact = "positive" if value > 0 else "negative"
        explanation.append(f"{feature} had a {impact} impact of {abs(value):.2f} on this recommendation")
    return explanation

def prepare_model_input(user_data):
    return {
        'credit_score': user_data['credit_score'],
        'income': user_data['income'],
        'debt_to_income_ratio': user_data['debt_to_income_ratio']
    }

def generate_loan_options(model_output, user_data):
    loan_types = ['Personal Loan', 'Home Loan', 'Auto Loan', 'Education Loan']
    personalized_loans = []
    
    for loan_type in loan_types:
        loan_option = {
            'loan_type': loan_type,
            'max_amount': calculate_max_amount(loan_type, model_output, user_data),
            'interest_rate': calculate_interest_rate(loan_type, model_output, user_data),
            'loan_term': suggest_loan_term(loan_type)
        }
        personalized_loans.append(loan_option)
    
    # Use LLM to generate personalized descriptions
    personalized_loans = add_llm_descriptions(personalized_loans, user_data)
    
    return personalized_loans

def calculate_max_amount(loan_type, model_output, user_data):
    # Implement logic to calculate max loan amount based on loan type and user data
    # This is a placeholder
    base_amount = user_data['income'] * 3
    if loan_type == 'Home Loan':
        return base_amount * 5
    elif loan_type == 'Auto Loan':
        return base_amount * 0.5
    elif loan_type == 'Education Loan':
        return base_amount * 2
    else:
        return base_amount

def calculate_interest_rate(loan_type, model_output, user_data):
    # Implement logic to calculate interest rate based on loan type and user data
    # This is a placeholder
    base_rate = 5.0
    if loan_type == 'Home Loan':
        return base_rate + 0.5
    elif loan_type == 'Auto Loan':
        return base_rate + 1.0
    elif loan_type == 'Education Loan':
        return base_rate + 1.5
    else:
        return base_rate + 2.0

def suggest_loan_term(loan_type):
    # Implement logic to suggest loan term based on loan type
    # This is a placeholder
    if loan_type == 'Home Loan':
        return 360  # 30 years
    elif loan_type == 'Auto Loan':
        return 60   # 5 years
    elif loan_type == 'Education Loan':
        return 120  # 10 years
    else:
        return 36   # 3 years

def add_llm_descriptions(personalized_loans, user_data):
    prompt = f"""
    User profile:
    Credit score: {user_data['credit_score']}
    Annual income: ${user_data['income']}
    Debt-to-income ratio: {user_data['debt_to_income_ratio']}
    
    Personalized loan options:
    {json.dumps(personalized_loans, indent=2)}
    
    Please provide a brief, personalized description for each loan option, highlighting its benefits and considerations for this specific user.
    """
    
    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-v2',
        body=json.dumps({
            'prompt': prompt,
            'max_tokens_to_sample': 1000,
            'temperature': 0.7,
            'top_p': 0.95,
        })
    )
    
    descriptions = json.loads(response['body'])['completion']
    
    # Parse the descriptions and add them to the personalized_loans
    # This part depends on the format of the LLM output
    # For simplicity, let's assume the LLM returns a JSON string
    parsed_descriptions = json.loads(descriptions)
    
    for loan, description in zip(personalized_loans, parsed_descriptions):
        loan['description'] = description
    
    return personalized_loans