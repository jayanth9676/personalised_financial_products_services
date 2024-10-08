import json
import boto3
import os
from aws_lambda_powertools import Metrics, Logger

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['USERS_TABLE_NAME'])

metrics = Metrics()
logger = Logger()

@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event, context):
    try:
        user_id = event['requestContext']['authorizer']['claims']['sub']
        user_data = get_user_data(user_id)
        
        return {
            'statusCode': 200,
            'body': json.dumps(user_data)
        }
    except Exception as e:
        logger.exception("Error fetching user data")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def get_user_data(user_id):
    response = table.get_item(Key={'user_id': user_id})
    return response.get('Item', {})