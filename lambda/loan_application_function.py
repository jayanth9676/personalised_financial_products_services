import json
import boto3
from botocore.exceptions import ClientError

sagemaker_runtime = boto3.client('sagemaker-runtime')
personalize = boto3.client('personalize-runtime')

def lambda_handler(event, context):
    try:
        loan_application = json.loads(event['body'])
        
        # Prepare the data for SageMaker endpoint
        sagemaker_input = prepare_sagemaker_input(loan_application)
        
        # Call SageMaker endpoint for prediction
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='your-sagemaker-endpoint-name',
            ContentType='application/json',
            Body=json.dumps(sagemaker_input)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        # Process the result and generate an offer if approved
        if result['approved']:
            offer = generate_offer(loan_application, result)
            return {
                'statusCode': 200,
                'body': json.dumps({'approved': True, 'offer': offer})
            }
        else:
            return {
                'statusCode': 200,
                'body': json.dumps({'approved': False, 'reason': result['reason']})
            }
    except ClientError as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def prepare_sagemaker_input(loan_application):
    # Transform loan_application data into the format expected by your SageMaker model
    # This will depend on how your model was trained
    return {
        'loan_amount': loan_application['loanAmount'],
        'loan_type': loan_application['loanType'],
        # Add other relevant features
    }

def generate_offer(loan_application, prediction_result):
    # Generate a personalized offer based on the loan application and prediction result
    # You might want to use AWS Personalize here for more sophisticated offer generation
    return {
        'interest_rate': prediction_result['recommended_interest_rate'],
        'loan_term': prediction_result['recommended_loan_term'],
        'monthly_payment': calculate_monthly_payment(loan_application['loanAmount'], prediction_result['recommended_interest_rate'], prediction_result['recommended_loan_term'])
    }

def calculate_monthly_payment(loan_amount, interest_rate, loan_term):
    # Implement the monthly payment calculation logic
    pass