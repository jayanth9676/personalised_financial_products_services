import json
import boto3
import os
import jwt
from datetime import datetime, timedelta
from aws_lambda_powertools import Metrics, Logger

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['USERS_TABLE_NAME'])

metrics = Metrics()
logger = Logger()

SECRET_KEY = os.environ['JWT_SECRET_KEY']

@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        username = body['username']
        password = body['password']
        
        user = authenticate_user(username, password)
        
        if user:
            token = generate_jwt_token(user)
            return {
                'statusCode': 200,
                'body': json.dumps({'token': token})
            }
        else:
            return {
                'statusCode': 401,
                'body': json.dumps({'error': 'Invalid credentials'})
            }
    except Exception as e:
        logger.exception("Error during login")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def authenticate_user(username, password):
    response = table.get_item(Key={'username': username})
    user = response.get('Item')
    if user and user['password'] == password:  # In production, use proper password hashing
        return user
    return None

def generate_jwt_token(user):
    payload = {
        'sub': user['user_id'],
        'username': user['username'],
        'exp': datetime.utcnow() + timedelta(days=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')