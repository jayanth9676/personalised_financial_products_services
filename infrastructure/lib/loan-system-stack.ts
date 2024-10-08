import * as cdk from '@aws-cdk/core';
import * as dynamodb from '@aws-cdk/aws-dynamodb';
import * as s3 from '@aws-cdk/aws-s3';
import * as lambda from '@aws-cdk/aws-lambda';
import * as apigateway from '@aws-cdk/aws-apigateway';
import * as cloudfront from '@aws-cdk/aws-cloudfront';
import * as origins from '@aws-cdk/aws-cloudfront-origins';
import * as amplify from '@aws-cdk/aws-amplify';
import * as sagemaker from '@aws-cdk/aws-sagemaker';
import * as events from '@aws-cdk/aws-events';
import * as targets from '@aws-cdk/aws-events-targets';
import * as iam from '@aws-cdk/aws-iam';

export class LoanSystemStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // DynamoDB table
    const loanApplicationsTable = new dynamodb.Table(this, 'LoanApplications', {
      partitionKey: { name: 'applicationId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    // S3 bucket for frontend hosting
    const websiteBucket = new s3.Bucket(this, 'WebsiteBucket', {
      websiteIndexDocument: 'index.html',
      publicReadAccess: true,
    });

    // CloudFront distribution
    const distribution = new cloudfront.Distribution(this, 'Distribution', {
      defaultBehavior: { origin: new origins.S3Origin(websiteBucket) },
    });

    // Lambda function
    const loanProcessingFunction = new lambda.Function(this, 'LoanProcessingFunction', {
      runtime: lambda.Runtime.PYTHON_3_8,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('lambda'),
      environment: {
        DYNAMODB_TABLE_NAME: loanApplicationsTable.tableName,
        SAGEMAKER_ENDPOINT: 'your-sagemaker-endpoint-name',
      },
    });

    // API Gateway
    const api = new apigateway.RestApi(this, 'LoanAPI');
    api.root.addMethod('POST', new apigateway.LambdaIntegration(loanProcessingFunction));

    // SageMaker model
    const model = new sagemaker.CfnModel(this, 'LoanPredictionModel', {
      executionRoleArn: 'your-sagemaker-role-arn',
      primaryContainer: {
        image: '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3',
        modelDataUrl: 's3://your-bucket/model.tar.gz',
      },
    });

    // Amplify App
    new amplify.App(this, 'LoanApp', {
      sourceCodeProvider: new amplify.GitHubSourceCodeProvider({
        owner: 'your-github-username',
        repository: 'your-repo-name',
        oauthToken: cdk.SecretValue.secretsManager('github-token'),
      }),
    });

    // Feedback table
    const feedbackTable = new dynamodb.Table(this, 'FeedbackTable', {
      partitionKey: { name: 'feedbackId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    // Feedback handling Lambda
    const feedbackHandlerFunction = new lambda.Function(this, 'FeedbackHandlerFunction', {
      runtime: lambda.Runtime.PYTHON_3_8,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('lambda/feedback_handler'),
      environment: {
        FEEDBACK_TABLE_NAME: feedbackTable.tableName,
      },
    });

    // Grant permissions
    feedbackTable.grantReadWriteData(feedbackHandlerFunction);

    // Add API Gateway endpoint for feedback
    api.root.addResource('feedback').addMethod('POST', new apigateway.LambdaIntegration(feedbackHandlerFunction));

    // Feedback analysis Lambda
    const feedbackAnalysisFunction = new lambda.Function(this, 'FeedbackAnalysisFunction', {
      runtime: lambda.Runtime.PYTHON_3_8,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('lambda/feedback_analysis'),
      environment: {
        FEEDBACK_TABLE_NAME: feedbackTable.tableName,
        KNOWLEDGE_BASE_TABLE_NAME: loanApplicationsTable.tableName, // Assuming we're using the same table for knowledge base
      },
      timeout: cdk.Duration.minutes(15),
    });

    // Grant permissions
    feedbackTable.grantReadData(feedbackAnalysisFunction);
    loanApplicationsTable.grantReadWriteData(feedbackAnalysisFunction);

    // Schedule feedback analysis to run daily
    new events.Rule(this, 'DailyFeedbackAnalysis', {
      schedule: events.Schedule.cron({ minute: '0', hour: '0' }), // Run at midnight UTC
      targets: [new targets.LambdaFunction(feedbackAnalysisFunction)],
    });

    // Output the resource names
    new cdk.CfnOutput(this, 'DynamoDBTableName', { value: loanApplicationsTable.tableName });
    new cdk.CfnOutput(this, 'WebsiteBucketName', { value: websiteBucket.bucketName });
    new cdk.CfnOutput(this, 'CloudFrontURL', { value: distribution.domainName });
    new cdk.CfnOutput(this, 'APIGatewayURL', { value: api.url });

    // Existing DynamoDB tables
    const userContextTable = new dynamodb.Table(this, 'UserContext', {
      partitionKey: { name: 'connectionId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    // New DynamoDB table for rate limiting
    const rateLimitTable = new dynamodb.Table(this, 'RateLimit', {
      partitionKey: { name: 'connectionId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'ttl',
    });

    // Lambda function for WebSocket handler
    const websocketHandler = new lambda.Function(this, 'WebSocketHandler', {
      runtime: lambda.Runtime.NODEJS_14_X,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('lambda/websocket-default'),
      environment: {
        USER_CONTEXT_TABLE: userContextTable.tableName,
        RATE_LIMIT_TABLE: rateLimitTable.tableName,
      },
    });

    // Grant permissions to the Lambda function
    userContextTable.grantReadWriteData(websocketHandler);
    rateLimitTable.grantReadWriteData(websocketHandler);

    // API Gateway WebSocket API
    const webSocketApi = new apigateway.WebSocketApi(this, 'WebSocketAPI', {
      connectRouteOptions: { integration: new apigateway.LambdaWebSocketIntegration({ handler: websocketHandler }) },
      disconnectRouteOptions: { integration: new apigateway.LambdaWebSocketIntegration({ handler: websocketHandler }) },
      defaultRouteOptions: { integration: new apigateway.LambdaWebSocketIntegration({ handler: websocketHandler }) },
    });

    new apigateway.WebSocketStage(this, 'WebSocketProdStage', {
      webSocketApi,
      stageName: 'prod',
      autoDeploy: true,
    });

    // Grant permissions to invoke the WebSocket API
    websocketHandler.addToRolePolicy(new iam.PolicyStatement({
      actions: ['execute-api:ManageConnections'],
      resources: [`arn:aws:execute-api:${this.region}:${this.account}:${webSocketApi.apiId}/*`],
    }));

    // Output the WebSocket URL
    new cdk.CfnOutput(this, 'WebSocketURL', {
      value: webSocketApi.apiEndpoint,
      description: 'WebSocket API Endpoint',
    });
  }
}