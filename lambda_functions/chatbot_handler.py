import json
import boto3
import os
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

patch_all()

bedrock = boto3.client('bedrock')
dynamodb = boto3.resource('dynamodb')

KNOWLEDGE_BASE_TABLE = os.environ['KNOWLEDGE_BASE_TABLE']

@xray_recorder.capture('chatbot_handler')
def lambda_handler(event, context):
    try:
        user_input = json.loads(event['body'])['message']
        
        # Retrieve relevant information from knowledge base
        table = dynamodb.Table(KNOWLEDGE_BASE_TABLE)
        knowledge_base_items = table.scan()['Items']
        
        # Construct prompt with retrieved information
        prompt = f"Based on the following information:\n"
        for item in knowledge_base_items:
            prompt += f"- {item['key']}: {item['value']}\n"
        prompt += f"\nUser question: {user_input}\nPlease provide a helpful response:"
        
        # Generate response using Bedrock
        response = bedrock.invoke_model(
            modelId='anthropic.claude-v2',
            body=json.dumps({
                'prompt': prompt,
                'max_tokens_to_sample': 300
            })
        )
        
        chatbot_response = json.loads(response['body'].read())['completion']
        
        return {
            'statusCode': 200,
            'body': json.dumps({'response': chatbot_response})
        }
    except Exception as e:
        print(f"Error in chatbot handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Internal server error'})
        }