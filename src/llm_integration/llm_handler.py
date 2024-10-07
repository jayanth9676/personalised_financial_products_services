import boto3
import json
from botocore.exceptions import ClientError

class LLMHandler:
    def __init__(self, model_id, region_name='us-west-2'):
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        self.model_id = model_id

    def generate_loan_offer(self, customer_profile, approval_status, model_explanation):
        prompt = self._create_prompt(customer_profile, approval_status, model_explanation)
        
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    "prompt": prompt,
                    "max_tokens_to_sample": 500,
                    "temperature": 0.7,
                    "top_p": 0.95,
                })
            )
            
            response_body = json.loads(response['body'].read())
            return self._parse_response(response_body['completion'])
        
        except ClientError as e:
            print(f"An error occurred: {e}")
            return None

    def _create_prompt(self, customer_profile, approval_status, model_explanation):
        prompt = f"""
        Given the following customer profile for a loan application:
        - Loan Type: {customer_profile['loan_type']}
        - Requested Amount: ${customer_profile['requested_amount']}
        - Credit Score: {customer_profile['credit_score']}
        - Annual Income: ${customer_profile['annual_income']}
        - Employment Status: {customer_profile['employment_status']}
        - Debt-to-Income Ratio: {customer_profile['dti_ratio']}

        The loan application has been {"APPROVED" if approval_status else "REJECTED"}.

        Model Explanation:
        {model_explanation}

        {"Generate a personalized loan offer" if approval_status else "Provide reasons for rejection and suggestions for improvement"}. The response should include:
        1. {"Approved loan amount" if approval_status else "Main reasons for rejection"}
        2. {"Interest rate" if approval_status else "Areas for improvement"}
        3. {"Loan term" if approval_status else "Suggested steps to increase approval chances"}
        4. {"Monthly payment" if approval_status else "Alternative loan options to consider"}
        5. A brief, empathetic explanation of the {"offer" if approval_status else "decision"}

        Format the response as a JSON object.
        """
        return prompt

    def _parse_response(self, response_text):
        try:
            offer = json.loads(response_text)
            return offer
        except json.JSONDecodeError:
            print("Error parsing LLM response")
            return None

# Usage
if __name__ == "__main__":
    llm_handler = LLMHandler('anthropic.claude-v2')
    customer_profile = {
        'loan_type': 'Personal',
        'requested_amount': 25000,
        'credit_score': 720,
        'annual_income': 75000,
        'employment_status': 'Employed',
        'dti_ratio': 0.3
    }
    approval_status = True
    model_explanation = "The most important factors for approval were high credit score and stable employment."
    offer = llm_handler.generate_loan_offer(customer_profile, approval_status, model_explanation)
    print(json.dumps(offer, indent=2))