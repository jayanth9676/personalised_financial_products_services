const AWS = require('aws-sdk');
const dynamodb = new AWS.DynamoDB.DocumentClient();

exports.handler = async (event) => {
    const offer = JSON.parse(event.body);
    const userId = event.requestContext.authorizer.claims.sub; // Assuming you're using Cognito for authentication

    const params = {
        TableName: process.env.OFFERS_TABLE_NAME,
        Item: {
            userId: userId,
            offerId: offer.id,
            status: 'accepted',
            ...offer
        }
    };

    try {
        await dynamodb.put(params).promise();
        return {
            statusCode: 200,
            body: JSON.stringify({ message: 'Offer accepted successfully' }),
        };
    } catch (error) {
        console.error('Error accepting offer:', error);
        return {
            statusCode: 500,
            body: JSON.stringify({ message: 'Failed to accept offer' }),
        };
    }
};