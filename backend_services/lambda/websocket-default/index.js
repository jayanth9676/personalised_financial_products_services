const AWS = require('aws-sdk');
const bedrock = new AWS.BedrockRuntime();
const dynamodb = new AWS.DynamoDB.DocumentClient();

exports.handler = async (event) => {
  console.log('WebSocket Default:', event);

  const { connectionId, domainName, stage } = event.requestContext;
  const body = JSON.parse(event.body);
  const { message } = body;

  const apigwManagementApi = new AWS.ApiGatewayManagementApi({
    apiVersion: '2018-11-29',
    endpoint: `${domainName}/${stage}`
  });

  try {
    // Retrieve user context from DynamoDB
    const userContext = await getUserContext(connectionId);

    const bedrockResponse = await bedrock.invokeModel({
      modelId: 'anthropic.claude-v2',
      contentType: 'application/json',
      accept: 'application/json',
      body: JSON.stringify({
        prompt: `Human: ${message}\n\nAssistant: As an AI assistant for a loan application system, I'll do my best to help you. Here's some context about your loan application: ${JSON.stringify(userContext)}. What would you like to know about our loan products or application process?`,
        max_tokens_to_sample: 300,
        temperature: 0.7,
        top_p: 0.95,
      }),
    }).promise();

    const aiResponse = JSON.parse(bedrockResponse.body).completion;

    // Update user context based on the conversation
    await updateUserContext(connectionId, message, aiResponse);

    await apigwManagementApi.postToConnection({
      ConnectionId: connectionId,
      Data: JSON.stringify({ event: 'ai_response', message: aiResponse })
    }).promise();

    return { statusCode: 200, body: 'Message sent' };
  } catch (error) {
    console.error('Error:', error);
    await apigwManagementApi.postToConnection({
      ConnectionId: connectionId,
      Data: JSON.stringify({ event: 'error', message: 'An error occurred while processing your request.' })
    }).promise();
    return { statusCode: 500, body: 'Error processing message' };
  }
};

async function getUserContext(connectionId) {
  // Implement DynamoDB get operation to retrieve user context
}

async function updateUserContext(connectionId, userMessage, aiResponse) {
  // Implement DynamoDB update operation to store conversation history
}