import boto3
import os
import random
import json
from aws_lambda_powertools import Logger, Metrics
from aws_lambda_powertools.metrics import MetricUnit

dynamodb = boto3.resource('dynamodb')
sagemaker_runtime = boto3.client('sagemaker-runtime')

USER_TABLE = os.environ['USER_TABLE']
MODEL_ENDPOINT_A = os.environ['MODEL_ENDPOINT_A']
MODEL_ENDPOINT_B = os.environ['MODEL_ENDPOINT_B']

logger = Logger()
metrics = Metrics()

def lambda_handler(event, context):
    try:
        user_id = event['requestContext']['authorizer']['claims']['sub']
        user_data = get_user_data(user_id)
        
        # Assign user to A or B group if not already assigned
        if 'ab_group' not in user_data:
            user_data['ab_group'] = 'A' if random.random() < 0.5 else 'B'
            update_user_data(user_id, {'ab_group': user_data['ab_group']})
        
        # Get loan recommendations based on the assigned group
        recommendations = get_recommendations(user_data)
        
        # Track metrics for the A/B test
        track_ab_test_metrics(user_data['ab_group'], recommendations)
        
        return {
            'statusCode': 200,
            'body': json.dumps(recommendations)
        }
    except Exception as e:
        logger.exception("Error in A/B test loan recommendations")
        return {
            'statusCode': 500,
            'body': str(e)
        }

def get_user_data(user_id):
    table = dynamodb.Table(USER_TABLE)
    response = table.get_item(Key={'user_id': user_id})
    return response['Item']

def update_user_data(user_id, update_data):
    table = dynamodb.Table(USER_TABLE)
    table.update_item(
        Key={'user_id': user_id},
        UpdateExpression="set " + ", ".join(f"#{k}=:{k}" for k in update_data),
        ExpressionAttributeNames={f"#{k}": k for k in update_data},
        ExpressionAttributeValues={f":{k}": v for k, v in update_data.items()}
    )

def get_recommendations(user_data):
    model_endpoint = MODEL_ENDPOINT_A if user_data['ab_group'] == 'A' else MODEL_ENDPOINT_B
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=model_endpoint,
        ContentType='application/json',
        Body=json.dumps(user_data)
    )
    return json.loads(response['Body'].read().decode())

def track_ab_test_metrics(ab_group, recommendations):
    metrics.add_metric(name=f"RecommendationsCount_{ab_group}", unit=MetricUnit.Count, value=1)
    metrics.add_metric(name=f"AverageInterestRate_{ab_group}", unit=MetricUnit.Percent, value=sum(r['interest_rate'] for r in recommendations) / len(recommendations))
    # Add more metrics as needed

# Update the API Gateway to use this Lambda function for loan recommendations