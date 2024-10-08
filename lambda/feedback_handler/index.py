import json
import boto3
import os
import uuid
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

patch_all()

dynamodb = boto3.resource('dynamodb')
FEEDBACK_TABLE_NAME = os.environ['FEEDBACK_TABLE_NAME']

@xray_recorder.capture('feedback_handler')
def handler(event, context):
    try:
        feedback_data = json.loads(event['body'])
        
        # Validate input data
        if not all(key in feedback_data for key in ['userId', 'responseId', 'rating', 'comment']):
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'Missing required fields'})
            }
        
        # Save feedback to DynamoDB
        table = dynamodb.Table(FEEDBACK_TABLE_NAME)
        feedback_id = str(uuid.uuid4())
        table.put_item(Item={
            'feedbackId': feedback_id,
            'userId': feedback_data['userId'],
            'responseId': feedback_data['responseId'],
            'rating': feedback_data['rating'],
            'comment': feedback_data['comment'],
            'timestamp': int(context.get_remaining_time_in_millis() / 1000)
        })
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Feedback submitted successfully', 'feedbackId': feedback_id})
        }
    except Exception as e:
        print(f"Error handling feedback: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Internal server error'})
        }