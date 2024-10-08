import json
import boto3
import os
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

patch_all()

bedrock = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')

KNOWLEDGE_BASE_TABLE = os.environ['KNOWLEDGE_BASE_TABLE']

@xray_recorder.capture('llm_rag_handler')
def lambda_handler(event, context):
    try:
        loan_data = json.loads(event['body'])
        
        # Retrieve relevant information from knowledge base
        knowledge_base_items = get_relevant_knowledge(loan_data)
        
        # Construct prompt with retrieved information
        prompt = construct_prompt(loan_data, knowledge_base_items)
        
        # Generate response using Bedrock
        response = bedrock.invoke_model(
            modelId='anthropic.claude-v2',
            body=json.dumps({
                'prompt': prompt,
                'max_tokens_to_sample': 500,
                'temperature': 0.7,
                'top_p': 0.95,
            })
        )
        
        llm_response = json.loads(response['body'])['completion']
        
        # Store the generated response for future reference
        store_response(loan_data, llm_response)
        
        return {
            'statusCode': 200,
            'body': json.dumps({'response': llm_response})
        }
    except Exception as e:
        print(f"Error in LLM & RAG handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Internal server error'})
        }

def get_relevant_knowledge(loan_data):
    table = dynamodb.Table(KNOWLEDGE_BASE_TABLE)
    # Implement logic to retrieve relevant knowledge based on loan_data
    # This is a simplified example
    response = table.scan()
    return response['Items']

def construct_prompt(loan_data, knowledge_base_items):
    context = "\n".join([item['value'] for item in knowledge_base_items])
    return f"""Human: I'm applying for a loan with the following details:
{json.dumps(loan_data, indent=2)}

Please provide a personalized loan offer and explanation based on this information.
"""