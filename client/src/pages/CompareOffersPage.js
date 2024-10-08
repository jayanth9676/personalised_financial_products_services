import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Container, Typography, Paper, Grid, Button } from '@mui/material';

function CompareOffersPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { originalOffer, updatedOffer } = location.state || {};

  if (!originalOffer || !updatedOffer) {
    navigate('/');
    return null;
  }

  return (
    <Container maxWidth="lg">
      <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Compare Loan Offers
        </Typography>
        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6">Original Offer</Typography>
            <Typography>Loan Amount: ${originalOffer.loan_amount}</Typography>
            <Typography>Loan Tenure: {originalOffer.loan_term} months</Typography>
            <Typography>Interest Rate: {originalOffer.interest_rate}%</Typography>
            <Typography>Monthly Payment: ${originalOffer.monthly_payment}</Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="h6">Updated Offer</Typography>
            <Typography>Loan Amount: ${updatedOffer.loan_amount}</Typography>
            <Typography>Loan Tenure: {updatedOffer.loan_term} months</Typography>
            <Typography>Interest Rate: {updatedOffer.interest_rate}%</Typography>
            <Typography>Monthly Payment: ${updatedOffer.monthly_payment}</Typography>
          </Grid>
        </Grid>
        <Box mt={3}>
          <Button variant="contained" color="primary" onClick={() => navigate('/personalized-offer')}>
            Back to Offer
          </Button>
        </Box>
      </Paper>
    </Container>
  );
}

export default CompareOffersPage;