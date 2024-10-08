import jwt

def lambda_handler(event, context):
    token = event['authorizationToken']
    
    try:
        # Verify and decode the JWT token
        decoded = jwt.decode(token, 'your-secret-key', algorithms=['HS256'])
        
        # Generate the IAM policy
        policy = generate_policy(decoded['sub'], 'Allow', event['methodArn'])
        
        return policy
    except jwt.ExpiredSignatureError:
        raise Exception('Expired token')
    except jwt.InvalidTokenError:
        raise Exception('Invalid token')

def generate_policy(principal_id, effect, resource):
    return {
        'principalId': principal_id,
        'policyDocument': {
            'Version': '2012-10-17',
            'Statement': [
                {
                    'Action': 'execute-api:Invoke',
                    'Effect': effect,
                    'Resource': resource
                }
            ]
        }
    }