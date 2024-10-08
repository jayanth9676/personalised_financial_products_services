import boto3
import os
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

patch_all()

dynamodb = boto3.resource('dynamodb')
FEEDBACK_TABLE_NAME = os.environ['FEEDBACK_TABLE_NAME']
KNOWLEDGE_BASE_TABLE_NAME = os.environ['KNOWLEDGE_BASE_TABLE_NAME']

@xray_recorder.capture('feedback_analysis')
def handler(event, context):
    try:
        feedback_table = dynamodb.Table(FEEDBACK_TABLE_NAME)
        knowledge_base_table = dynamodb.Table(KNOWLEDGE_BASE_TABLE_NAME)
        
        # Scan all feedback (in a production system, you'd want to paginate this)
        feedback_items = feedback_table.scan()['Items']
        
        # Analyze feedback (this is a simplified example)
        low_rated_responses = [item for item in feedback_items if item['rating'] < 3]
        
        # Update knowledge base based on feedback
        for response in low_rated_responses:
            # In a real system, you'd want to do more sophisticated analysis here
            knowledge_base_table.put_item(Item={
                'key': f"improvement_needed_{response['responseId']}",
                'value': f"Response {response['responseId']} needs improvement. User comment: {response['comment']}"
            })
        
        print(f"Analyzed {len(feedback_items)} feedback items and updated knowledge base.")
        return {
            'statusCode': 200,
            'body': f"Analyzed {len(feedback_items)} feedback items and updated knowledge base."
        }
    except Exception as e:
        print(f"Error analyzing feedback: {str(e)}")
        return {
            'statusCode': 500,
            'body': f"Error analyzing feedback: {str(e)}"
        }