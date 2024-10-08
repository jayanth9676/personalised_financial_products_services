const AWS = require('aws-sdk');
const dynamodb = new AWS.DynamoDB.DocumentClient();

const RATE_LIMIT_WINDOW = 60000; // 1 minute
const MAX_REQUESTS = 10; // 10 requests per minute

class RateLimitExceededError extends Error {
  constructor(message) {
    super(message);
    this.name = 'RateLimitExceededError';
  }
}

async function rateLimit(connectionId, tableName) {
  const now = Date.now();
  const windowStart = now - RATE_LIMIT_WINDOW;

  const params = {
    TableName: tableName,
    Key: { connectionId },
    UpdateExpression: 'SET requests = list_append(if_not_exists(requests, :empty_list), :new_request), ttl = :ttl',
    ExpressionAttributeValues: {
      ':empty_list': [],
      ':new_request': [now],
      ':window_start': windowStart,
      ':ttl': Math.floor(Date.now() / 1000) + 86400, // TTL of 24 hours
    },
    ReturnValues: 'ALL_NEW',
  };

  try {
    const result = await dynamodb.update(params).promise();
    const requests = result.Attributes.requests.filter(timestamp => timestamp >= windowStart);

    if (requests.length > MAX_REQUESTS) {
      throw new RateLimitExceededError('Rate limit exceeded');
    }

    // Clean up old requests
    if (requests.length < result.Attributes.requests.length) {
      await dynamodb.update({
        TableName: tableName,
        Key: { connectionId },
        UpdateExpression: 'SET requests = :requests',
        ExpressionAttributeValues: { ':requests': requests },
      }).promise();
    }
  } catch (error) {
    if (error.name === 'RateLimitExceededError') {
      throw error;
    }
    console.error('Error checking rate limit:', error);
    // If there's an error checking the rate limit, we'll allow the request to proceed
  }
}

module.exports = rateLimit;