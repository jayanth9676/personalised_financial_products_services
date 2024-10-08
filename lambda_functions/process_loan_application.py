import json
import boto3
import os
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

patch_all()

dynamodb = boto3.resource('dynamodb')
sagemaker_runtime = boto3.client('sagemaker-runtime')
bedrock = boto3.client('bedrock-runtime')

TABLE_NAME = os.environ['DYNAMODB_TABLE_NAME']
SAGEMAKER_ENDPOINT = os.environ['SAGEMAKER_ENDPOINT']

@xray_recorder.capture('process_loan_application')
def lambda_handler(event, context):
    try:
        loan_data = json.loads(event['body'])
        
        # Validate input data
        if not all(key in loan_data for key in ['loanAmount', 'loanPurpose', 'annualIncome', 'employmentStatus', 'creditScore']):
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'Missing required fields'})
            }
        
        # Call SageMaker endpoint for loan approval prediction
        sagemaker_response = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType='application/json',
            Body=json.dumps(loan_data)
        )
        prediction = json.loads(sagemaker_response['Body'].read().decode())
        
        # Generate personalized offer using Bedrock
        bedrock_response = bedrock.invoke_model(
            modelId='anthropic.claude-v2',
            body=json.dumps({
                'prompt': f"Generate a personalized loan offer based on: {json.dumps(loan_data)}. Prediction: {prediction['approved']}",
                'max_tokens_to_sample': 500
            })
        )
        offer = json.loads(bedrock_response['body'])['completion']
        
        # Save application to DynamoDB
        table = dynamodb.Table(TABLE_NAME)
        table.put_item(Item={
            'applicationId': context.aws_request_id,
            'loanData': loan_data,
            'prediction': prediction,
            'offer': offer
        })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'applicationId': context.aws_request_id,
                'approved': prediction['approved'],
                'offer': offer
            })
        }
    except Exception as e:
        print(f"Error processing loan application: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Internal server error'})
        }