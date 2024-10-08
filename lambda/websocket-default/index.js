const { BedrockRuntimeClient, InvokeModelCommand } = require("@aws-sdk/client-bedrock-runtime");
const { DynamoDBClient } = require("@aws-sdk/client-dynamodb");
const { DynamoDBDocumentClient, GetCommand, UpdateCommand } = require("@aws-sdk/lib-dynamodb");
const { ApiGatewayManagementApiClient, PostToConnectionCommand } = require("@aws-sdk/client-apigatewaymanagementapi");
const rateLimit = require('./rateLimit');

const bedrock = new BedrockRuntimeClient();
const ddbClient = new DynamoDBClient();
const ddbDocClient = DynamoDBDocumentClient.from(ddbClient);

const USER_CONTEXT_TABLE = process.env.USER_CONTEXT_TABLE;
const RATE_LIMIT_TABLE = process.env.RATE_LIMIT_TABLE;

exports.handler = async (event) => {
  console.log('WebSocket Default:', event);

  const { connectionId, domainName, stage } = event.requestContext;
  let body;
  try {
    body = JSON.parse(event.body);
  } catch (error) {
    console.error('Error parsing event body:', error);
    return { statusCode: 400, body: 'Invalid request body' };
  }

  const { message } = body;

  if (!message || typeof message !== 'string' || message.length > 1000) {
    return { statusCode: 400, body: 'Invalid message' };
  }

  const apigwManagementApi = new ApiGatewayManagementApiClient({
    apiVersion: '2018-11-29',
    endpoint: `https://${domainName}/${stage}`
  });

  try {
    // Check rate limit
    await rateLimit(connectionId, RATE_LIMIT_TABLE);

    // Retrieve user context from DynamoDB
    const userContext = await getUserContext(connectionId);

    const bedrockResponse = await bedrock.send(new InvokeModelCommand({
      modelId: 'anthropic.claude-v2',
      contentType: 'application/json',
      accept: 'application/json',
      body: JSON.stringify({
        prompt: generatePrompt(message, userContext),
        max_tokens_to_sample: 300,
        temperature: 0.7,
        top_p: 0.95,
      }),
    }));

    const aiResponse = JSON.parse(new TextDecoder().decode(bedrockResponse.body)).completion;

    // Update user context based on the conversation
    await updateUserContext(connectionId, message, aiResponse);

    await apigwManagementApi.send(new PostToConnectionCommand({
      ConnectionId: connectionId,
      Data: JSON.stringify({ event: 'ai_response', message: aiResponse })
    }));

    return { statusCode: 200, body: 'Message sent' };
  } catch (error) {
    console.error('Error:', error);
    if (error.name === 'RateLimitExceededError') {
      return { statusCode: 429, body: 'Rate limit exceeded' };
    }
    await apigwManagementApi.send(new PostToConnectionCommand({
      ConnectionId: connectionId,
      Data: JSON.stringify({ event: 'error', message: 'An error occurred while processing your request.' })
    }));
    return { statusCode: 500, body: 'Error processing message' };
  }
};

function generatePrompt(message, userContext) {
  const contextString = Object.entries(userContext)
    .map(([key, value]) => `${key}: ${value}`)
    .join(', ');
  return `Human: ${message}\n\nAssistant: As an AI assistant for a loan application system, I'll do my best to help you. Here's some context about your loan application: ${contextString}. What would you like to know about our loan products or application process?`;
}

async function getUserContext(connectionId) {
  const params = {
    TableName: USER_CONTEXT_TABLE,
    Key: { connectionId }
  };

  try {
    const result = await ddbDocClient.send(new GetCommand(params));
    return result.Item || {};
  } catch (error) {
    console.error('Error retrieving user context:', error);
    return {};
  }
}

async function updateUserContext(connectionId, userMessage, aiResponse) {
  const params = {
    TableName: USER_CONTEXT_TABLE,
    Key: { connectionId },
    UpdateExpression: 'SET lastUserMessage = :userMessage, lastAIResponse = :aiResponse, updatedAt = :timestamp, messageCount = if_not_exists(messageCount, :zero) + :one',
    ExpressionAttributeValues: {
      ':userMessage': userMessage,
      ':aiResponse': aiResponse,
      ':timestamp': new Date().toISOString(),
      ':zero': 0,
      ':one': 1
    },
    ReturnValues: 'NONE'
  };

  try {
    await ddbDocClient.send(new UpdateCommand(params));
  } catch (error) {
    console.error('Error updating user context:', error);
  }
}