import json
import boto3
import joblib
import numpy as np
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    # Load the model and preprocessor
    s3 = boto3.client('s3')
    bucket_name = 'your-model-bucket'
    model_key = 'loan_approval_model.joblib'
    preprocessor_key = 'preprocessor.joblib'
    
    try:
        model = joblib.load(s3.get_object(Bucket=bucket_name, Key=model_key)['Body'].read())
        preprocessor = joblib.load(s3.get_object(Bucket=bucket_name, Key=preprocessor_key)['Body'].read())
    except ClientError as e:
        return {
            'statusCode': 500,
            'body': json.dumps('Error loading model or preprocessor')
        }
    
    # Parse the input data
    loan_application = json.loads(event['body'])
    
    # Preprocess the input data
    input_data = preprocess_input(loan_application, preprocessor)
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    # Get model explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    feature_importance = dict(zip(preprocessor['selected_features'], np.abs(shap_values[0]).mean(0)))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Generate explanation
    explanation = "The top factors influencing the decision were: "
    explanation += ", ".join([f"{feature} (importance: {importance:.2f})" for feature, importance in top_features])
    
    # Use LLM to generate personalized response
    llm_handler = LLMHandler('anthropic.claude-v2')
    llm_response = llm_handler.generate_loan_offer(loan_application, prediction[0], explanation)
    
    response = {
        'approval_status': bool(prediction[0]),
        'approval_probability': float(prediction_proba),
        'explanation': explanation,
        'llm_response': llm_response
    }
    
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }

def preprocess_input(loan_application, preprocessor):
    # Convert loan application to DataFrame
    input_df = pd.DataFrame([loan_application])
    
    # Apply preprocessing steps
    categorical_columns = input_df.select_dtypes(include=['object']).columns
    numeric_columns = input_df.select_dtypes(include=['int64', 'float64']).columns
    
    input_df[numeric_columns] = preprocessor['scaler'].transform(input_df[numeric_columns])
    encoded_categorical = preprocessor['encoder'].transform(input_df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_categorical, columns=preprocessor['encoder'].get_feature_names_out(categorical_columns))
    
    processed_input = pd.concat([input_df.drop(columns=categorical_columns), encoded_df], axis=1)
    
    # Select only the features used by the model
    return processed_input[preprocessor['selected_features']]