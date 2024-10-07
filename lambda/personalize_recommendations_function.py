import json
import boto3
from botocore.exceptions import ClientError

personalize_runtime = boto3.client('personalize-runtime')

def lambda_handler(event, context):
    try:
        user_id = event['requestContext']['authorizer']['claims']['sub']  # Assuming you're using Cognito for authentication
        
        response = personalize_runtime.get_recommendations(
            campaignArn='your-personalize-campaign-arn',
            userId=user_id
        )
        
        recommendations = [
            {
                'type': item['itemId'],
                'suggestedAmount': float(item['metadata']['suggestedAmount']),
                'estimatedAPR': float(item['metadata']['estimatedAPR'])
            }
            for item in response['itemList']
        ]
        
        return {
            'statusCode': 200,
            'body': json.dumps(recommendations)
        }
    except ClientError as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }