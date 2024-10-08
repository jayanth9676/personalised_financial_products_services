import json
import boto3
import os
from aws_lambda_powertools import Metrics, Logger

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['OFFERS_TABLE_NAME'])

metrics = Metrics()
logger = Logger()

@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event, context):
    try:
        user_id = event['requestContext']['authorizer']['claims']['sub']
        offer = json.loads(event.get('body', '{}'))
        
        if not offer or 'id' not in offer:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid offer data'})
            }
        
        response = table.update_item(
            Key={'user_id': user_id, 'offer_id': offer['id']},
            UpdateExpression="set #status = :s, accepted_at = :t",
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':s': 'accepted',
                ':t': datetime.now().isoformat()
            },
            ReturnValues="UPDATED_NEW"
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Offer accepted successfully'})
        }
    except Exception as e:
        logger.exception("Error accepting offer")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }