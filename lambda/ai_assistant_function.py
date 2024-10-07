import json
import boto3
from botocore.exceptions import ClientError

bedrock = boto3.client('bedrock-runtime')
kendra = boto3.client('kendra')

def lambda_handler(event, context):
    try:
        message = json.loads(event['body'])['message']
        
        # First, try to find an answer using Kendra
        kendra_response = kendra.query(
            IndexId='your-kendra-index-id',
            QueryText=message
        )
        
        if kendra_response['ResultItems']:
            answer = kendra_response['ResultItems'][0]['DocumentExcerpt']['Text']
        else:
            # If Kendra doesn't have an answer, use Bedrock
            bedrock_response = bedrock.invoke_model(
                modelId='anthropic.claude-v2',
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    "prompt": f"Human: {message}\nAI:",
                    "max_tokens_to_sample": 300
                })
            )
            answer = json.loads(bedrock_response['body'].read())['completion']
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': answer})
        }
    except ClientError as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }