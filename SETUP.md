# Detailed AWS Setup Guide for AI-Powered Loan Recommendation System

## 1. AWS Account Setup
1.1. Create an AWS account at aws.amazon.com if you don't have one.
1.2. Set up Multi-Factor Authentication (MFA) for enhanced security.
1.3. Create an IAM user with appropriate permissions for development.

## 2. Local Development Environment Setup
2.1. Install AWS CLI:
   - Download from https://aws.amazon.com/cli/
   - Run `aws configure` and enter your AWS access key, secret key, and preferred region.
2.2. Install Node.js and npm.
2.3. Install AWS CDK: `npm install -g aws-cdk`

## 3. Project Initialization
3.1. Clone the repository: `git clone [your-repo-url]`
3.2. Navigate to the project directory: `cd loan-recommendation-system`
3.3. Install dependencies: `npm install`

## 4. AWS Services Configuration

### 4.1. Amazon DynamoDB
4.1.1. The CDK stack will automatically create the required DynamoDB tables:
   - LoanApplications
   - UserContext
   - RateLimit
   - FeedbackTable

### 4.2. Amazon S3
4.2.1. The CDK stack will create an S3 bucket for frontend hosting.

### 4.3. AWS Lambda
4.3.1. Lambda functions will be automatically created and configured by the CDK stack:
   - LoanProcessingFunction
   - WebSocketHandler
   - FeedbackHandlerFunction
   - FeedbackAnalysisFunction

### 4.4. Amazon API Gateway
4.4.1. REST API and WebSocket API will be set up automatically by the CDK stack.

### 4.5. Amazon SageMaker
4.5.1. Set up a SageMaker notebook instance:
   - Navigate to SageMaker in the AWS Console
   - Create a new notebook instance
   - Upload and run the `train_and_deploy.py` script

### 4.6. AWS Bedrock
4.6.1. Enable AWS Bedrock in your AWS account.
4.6.2. Set up access to the Claude model.

### 4.7. Amazon CloudFront
4.7.1. A CloudFront distribution will be created automatically by the CDK stack.

### 4.8. AWS Amplify
4.8.1. The Amplify app will be created by the CDK stack.
4.8.2. Configure the Amplify app with your GitHub repository for CI/CD.

## 5. Deployment
5.1. Build the CDK stack: `cdk synth`
5.2. Deploy the stack: `cdk deploy`

## 6. Post-Deployment Configuration
6.1. Update the `aws-exports.js` file in the frontend with the correct API endpoints and WebSocket URL.
6.2. Update Lambda environment variables with the deployed SageMaker endpoint name.

## 7. Frontend Deployment
7.1. Navigate to the frontend directory: `cd frontend`
7.2. Install dependencies: `npm install`
7.3. Build the React application: `npm run build`
7.4. The Amplify app will automatically deploy changes pushed to the connected GitHub repository.

## 8. Testing
8.1. Test the deployed APIs using Postman or curl.
8.2. Test the WebSocket connection using a WebSocket client.
8.3. Access the frontend through the CloudFront URL and test the full application flow.

## 9. Monitoring and Logging
9.1. Set up CloudWatch dashboards and alarms as defined in the MonitoringStack.
9.2. Review CloudWatch Logs for Lambda functions and API Gateway.

## 10. Cleanup
10.1. To remove all created resources, run: `cdk destroy`

Remember to replace placeholder values (like '[your-repo-url]', 'your-sagemaker-endpoint-name', etc.) with your actual values throughout the setup process.