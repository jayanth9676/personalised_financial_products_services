import React, { useEffect, useState } from 'react';
import { Container, Typography, Grid, Card, CardContent, Button, CircularProgress } from '@mui/material';
import { Link } from 'react-router-dom';
// import { API } from 'aws-amplify';
import { useSnackbar } from 'notistack';

function LoanRecommendationsPage() {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const { enqueueSnackbar } = useSnackbar();

  useEffect(() => {
    fetchRecommendations();
  }, []);

  const fetchRecommendations = async () => {
    try {
      // Simulating API call
      const mockRecommendations = [
        { type: 'Personal', suggestedAmount: 10000, estimatedAPR: 5.5 },
        { type: 'Auto', suggestedAmount: 25000, estimatedAPR: 4.5 },
        { type: 'Home', suggestedAmount: 200000, estimatedAPR: 3.5 },
      ];
      setRecommendations(mockRecommendations);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      enqueueSnackbar('Failed to fetch loan recommendations', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="md" sx={{ mt: 4, textAlign: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h3" gutterBottom align="center">
        Personalized Loan Recommendations
      </Typography>
      <Grid container spacing={3}>
        {recommendations.map((loan, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  {loan.type} Loan
                </Typography>
                <Typography variant="body1">
                  Suggested Amount: ${loan.suggestedAmount.toLocaleString()}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Estimated APR: {loan.estimatedAPR}%
                </Typography>
                <Button
                  component={Link}
                  to={`/apply/${loan.type.toLowerCase()}`}
                  variant="contained"
                  color="primary"
                  fullWidth
                  sx={{ mt: 2 }}
                >
                  Apply Now
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}

export default LoanRecommendationsPage;