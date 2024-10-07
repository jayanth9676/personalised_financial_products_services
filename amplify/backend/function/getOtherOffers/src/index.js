const AWS = require('aws-sdk');
const dynamodb = new AWS.DynamoDB.DocumentClient();

exports.handler = async (event) => {
    const userId = event.requestContext.authorizer.claims.sub;

    const params = {
        TableName: process.env.OFFERS_TABLE_NAME,
        IndexName: 'UserIdIndex',
        KeyConditionExpression: 'userId = :userId',
        ExpressionAttributeValues: {
            ':userId': userId
        }
    };

    try {
        const result = await dynamodb.query(params).promise();
        return {
            statusCode: 200,
            body: JSON.stringify(result.Items),
        };
    } catch (error) {
        console.error('Error fetching other offers:', error);
        return {
            statusCode: 500,
            body: JSON.stringify({ message: 'Failed to fetch other offers' }),
        };
    }
};