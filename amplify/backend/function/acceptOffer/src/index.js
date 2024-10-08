const AWS = require('aws-sdk');
const dynamodb = new AWS.DynamoDB.DocumentClient();

exports.handler = async (event) => {
    try {
        if (!process.env.OFFERS_TABLE_NAME) {
            throw new Error('OFFERS_TABLE_NAME environment variable is not set');
        }

        const offer = JSON.parse(event.body);
        const userId = event.requestContext.authorizer.claims.sub;

        if (!offer || !offer.id) {
            throw new Error('Invalid offer data');
        }

        const params = {
            TableName: process.env.OFFERS_TABLE_NAME,
            Item: {
                userId: userId,
                offerId: offer.id,
                status: 'accepted',
                updatedAt: new Date().toISOString(),
                // Only update specific fields instead of overwriting the entire object
                ...(offer.price && { price: offer.price }),
                ...(offer.details && { details: offer.details }),
                ...(offer.explanation && { explanation: offer.explanation }),
                // Add other specific fields as needed
                acceptedAt: new Date().toISOString()
            },
            ConditionExpression: 'attribute_exists(offerId)'
        };

        // Add a timestamp to the offer
        params.Item.acceptedAt = new Date().toISOString();

        await dynamodb.put(params).promise();

        // After accepting the offer, you might want to trigger some follow-up actions
        // For example, sending a confirmation email or updating other related records

        return {
            statusCode: 200,
            headers: {
                'Access-Control-Allow-Origin': '*', // Adjust this for production
                'Access-Control-Allow-Credentials': true,
            },
            body: JSON.stringify({ 
                message: 'Offer accepted successfully',
                explanation: offer.explanation
            }),
        };
    } catch (error) {
        console.error('Error accepting offer:', error);

        let statusCode = 500;
        let errorMessage = 'Failed to accept offer';

        if (error.name === 'ConditionalCheckFailedException') {
            statusCode = 404;
            errorMessage = 'Offer not found';
        } else if (error.message === 'Invalid offer data') {
            statusCode = 400;
            errorMessage = error.message;
        } else if (error.message === 'OFFERS_TABLE_NAME environment variable is not set') {
            statusCode = 500;
            errorMessage = 'Server configuration error';
        }

        return {
            statusCode: statusCode,
            headers: {
                'Access-Control-Allow-Origin': '*', // Adjust this for production
                'Access-Control-Allow-Credentials': true,
            },
            body: JSON.stringify({ message: errorMessage }),
        };
    }
};