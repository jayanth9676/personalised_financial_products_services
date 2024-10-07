import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Container, Typography, Grid, Button, Card, CardContent } from '@mui/material';
// import { StyledCard } from '../components/StyledComponents';
// import { API } from 'aws-amplify';
import CircularProgress from '@mui/material/CircularProgress';

function CompareOffersPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [currentOffer, setCurrentOffer] = useState(location.state?.currentOffer || null);
  const [otherOffers, setOtherOffers] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchOtherOffers();
  }, []);

  const fetchOtherOffers = async () => {
    try {
      // Simulated API call
      const mockOffers = [
        { id: 1, loanAmount: 10000, interestRate: 5.5, loanTerm: 3, monthlyPayment: 300 },
        { id: 2, loanAmount: 15000, interestRate: 6.0, loanTerm: 5, monthlyPayment: 290 },
      ];
      setOtherOffers(mockOffers);
    } catch (error) {
      console.error('Error fetching other offers:', error);
      setError('Failed to fetch other offers. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderOfferCard = (offer, isCurrentOffer = false) => (
    <Grid item xs={12} md={6} key={offer.id}>
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            {isCurrentOffer ? 'Current Offer' : 'Alternative Offer'}
          </Typography>
          <Typography>Loan Amount: ${offer.loanAmount.toLocaleString()}</Typography>
          <Typography>Interest Rate: {offer.interestRate.toFixed(2)}%</Typography>
          <Typography>Loan Term: {offer.loanTerm} years</Typography>
          <Typography>Monthly Payment: ${offer.monthlyPayment.toFixed(2)}</Typography>
          {!isCurrentOffer && (
            <Button
              variant="contained"
              color="primary"
              onClick={() => navigate('/offer', { state: { offer } })}
            >
              Select This Offer
            </Button>
          )}
        </CardContent>
      </Card>
    </Grid>
  );

  return (
    <Container maxWidth="lg">
      <Typography variant="h3" gutterBottom align="center">
        Compare Loan Offers
      </Typography>
      <Grid container spacing={4}>
        {currentOffer && renderOfferCard(currentOffer, true)}
        {otherOffers.map((offer) => renderOfferCard(offer))}
      </Grid>
      {isLoading && <CircularProgress />}
      {error && <Typography color="error">{error}</Typography>}
    </Container>
  );
}

export default CompareOffersPage;