import React, { useEffect, useState } from 'react';
import { Container, Typography, Grid, Card, CardContent, CardActions, Button, CircularProgress } from '@mui/material';
import { Link } from 'react-router-dom';
import { API } from 'aws-amplify';

function HomePage() {
  const [personalizedLoans, setPersonalizedLoans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchPersonalizedLoans();
  }, []);

  const fetchPersonalizedLoans = async () => {
    try {
      setLoading(true);
      const response = await API.get('LoanAPI', '/personalized-loans');
      setPersonalizedLoans(response);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching personalized loans:', err);
      setError('Failed to load personalized loan options. Please try again later.');
      setLoading(false);
    }
  };

  if (loading) {
    return <CircularProgress />;
  }

  if (error) {
    return <Typography color="error">{error}</Typography>;
  }

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom>
        Welcome to AI-Powered Loan Recommendations
      </Typography>
      <Grid container spacing={4}>
        {personalizedLoans.map((loan, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card>
              <CardContent>
                <Typography variant="h5" component="h2">
                  {loan.loan_type}
                </Typography>
                <Typography color="textSecondary">
                  Max Amount: ${loan.max_amount.toLocaleString()}
                </Typography>
                <Typography color="textSecondary">
                  Interest Rate: {loan.interest_rate}%
                </Typography>
                <Typography color="textSecondary">
                  Loan Term: {loan.loan_term} months
                </Typography>
                <Typography variant="body2" component="p">
                  {loan.description}
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="small" component={Link} to={`/apply/${loan.loan_type}`}>
                  Apply Now
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}

export default HomePage;