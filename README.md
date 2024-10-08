# Personalized Loan Application System

A fully functional loan application system that provides personalized loan recommendations based on user data and allows real-time adjustments of loan parameters.

## Features

- User authentication and authorization
- Personalized loan recommendations
- Real-time loan parameter adjustments
- Explainable AI for loan decisions
- Integration with AWS services (Lambda, DynamoDB, SageMaker, Bedrock)

## Tech Stack

- Frontend: React.js
- Backend: AWS Lambda (Node.js, Python)
- Database: AWS DynamoDB
- Machine Learning: AWS SageMaker
- Natural Language Processing: AWS Bedrock
- API: AWS API Gateway
- Authentication: JSON Web Tokens (JWT)

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-application-system.git
   cd loan-application-system
   ```

2. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

3. Set up AWS CLI and configure your credentials:
   ```bash
   aws configure
   ```

4. Deploy the backend:
   ```bash
   cd ../backend
   sam build
   sam deploy --guided
   ```

5. Update the frontend configuration:
   - Copy `frontend/src/config.example.js` to `frontend/src/config.js`
   - Update the API endpoint in `config.js` with the deployed API Gateway URL

6. Start the frontend application:
   ```bash
   cd ../frontend
   npm start
   ```

## Project Structure

- `/frontend`: React.js frontend application
- `/backend`: AWS SAM template and Lambda functions
- `/docs`: Additional documentation

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.