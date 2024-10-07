import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import { Container, Typography, Grid, Slider, Button, Box, Tooltip, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Card, CardContent } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import BookmarkIcon from '@mui/icons-material/Bookmark';
import LoanApplicationStepper from '../components/LoanApplicationStepper';

function OfferPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [offer, setOffer] = useState(location.state?.offer || null);
  const [originalOffer, setOriginalOffer] = useState(location.state?.offer || null);
  const [openConfirmDialog, setOpenConfirmDialog] = useState(false);
  const { enqueueSnackbar } = useSnackbar();
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!offer) {
      navigate('/recommendations');
      enqueueSnackbar('No offer available. Redirecting to recommendations.', { variant: 'info' });
    }
  }, [offer, navigate, enqueueSnackbar]);

  const handleAmountChange = (event, newValue) => {
    setOffer(prevOffer => ({
      ...prevOffer,
      loanAmount: newValue,
      monthlyPayment: calculateMonthlyPayment(newValue, prevOffer.loanTerm, prevOffer.interestRate)
    }));
  };

  const handleTermChange = (event, newValue) => {
    setOffer(prevOffer => ({
      ...prevOffer,
      loanTerm: newValue,
      monthlyPayment: calculateMonthlyPayment(prevOffer.loanAmount, newValue, prevOffer.interestRate)
    }));
  };

  const calculateMonthlyPayment = (amount, term, rate) => {
    const monthlyRate = rate / 100 / 12;
    const numberOfPayments = term * 12;
    return (amount * monthlyRate * Math.pow(1 + monthlyRate, numberOfPayments)) / (Math.pow(1 + monthlyRate, numberOfPayments) - 1);
  };

  const handleAcceptOffer = async () => {
    setOpenConfirmDialog(true);
  };

  const confirmAcceptOffer = async () => {
    setIsLoading(true);
    try {
      // Commented out AWS Amplify API call
      // const response = await API.post('loanAPI', '/accept-offer', { body: offer });
      console.log('Offer accepted:', offer);
      enqueueSnackbar('Offer accepted successfully!', { variant: 'success' });
      navigate('/dashboard');
    } catch (error) {
      console.error('Error accepting offer:', error);
      enqueueSnackbar('Failed to accept offer. Please try again.', { variant: 'error' });
    }
    setIsLoading(false);
    setOpenConfirmDialog(false);
  };

  const handleResetOffer = () => {
    setOffer(originalOffer);
    enqueueSnackbar('Offer reset to original values.', { variant: 'info' });
  };

  const handleSaveForLater = () => {
    // TODO: Implement save for later functionality
    enqueueSnackbar('Offer saved for later.', { variant: 'success' });
  };

  const handleCompareOffers = () => {
    navigate('/compare-offers', { state: { currentOffer: offer } });
  };

  const calculateAmortizationSchedule = () => {
    const monthlyRate = offer.interestRate / 100 / 12;
    const numberOfPayments = offer.loanTerm * 12;
    const monthlyPayment = (offer.loanAmount * monthlyRate * Math.pow(1 + monthlyRate, numberOfPayments)) / (Math.pow(1 + monthlyRate, numberOfPayments) - 1);

    let balance = offer.loanAmount;
    const schedule = [];

    for (let month = 1; month <= numberOfPayments; month++) {
      const interest = balance * monthlyRate;
      const principal = monthlyPayment - interest;
      balance -= principal;

      if (month % 12 === 0) {
        schedule.push({
          year: month / 12,
          balance: balance > 0 ? balance : 0,
        });
      }
    }

    return schedule;
  };

  if (!offer) return null;

  const amortizationSchedule = calculateAmortizationSchedule();
  const isOfferChanged = JSON.stringify(offer) !== JSON.stringify(originalOffer);

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <LoanApplicationStepper activeStep={2} />
      <Typography variant="h3" gutterBottom align="center">
        Your Personalized Loan Offer
      </Typography>
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Loan Amount
              </Typography>
              <Slider
                value={offer.loanAmount}
                onChange={handleAmountChange}
                min={offer.minAmount}
                max={offer.maxAmount}
                step={1000}
                marks
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `$${value.toLocaleString()}`}
                aria-label="Loan amount"
              />
              <Typography variant="h4" align="center" sx={{ mt: 2 }}>
                ${offer.loanAmount.toLocaleString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Loan Term
              </Typography>
              <Slider
                value={offer.loanTerm}
                onChange={handleTermChange}
                min={offer.minTerm}
                max={offer.maxTerm}
                step={1}
                marks
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value} years`}
                aria-label="Loan term"
              />
              <Typography variant="h4" align="center" sx={{ mt: 2 }}>
                {offer.loanTerm} years
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Offer Summary
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body1">Interest Rate:</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body1" align="right">
                    {offer.interestRate.toFixed(2)}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body1">Monthly Payment:</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography
                    variant="body1"
                    align="right"
                    sx={{
                      color: isOfferChanged
                        ? offer.monthlyPayment > originalOffer.monthlyPayment
                          ? 'error.main'
                          : 'success.main'
                        : 'inherit'
                    }}
                  >
                    ${offer.monthlyPayment.toFixed(2)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body1">
                    Total Interest:
                    <Tooltip title="The total amount of interest you'll pay over the life of the loan">
                      <InfoIcon fontSize="small" sx={{ ml: 1, verticalAlign: 'middle' }} />
                    </Tooltip>
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body1" align="right">
                    ${((offer.monthlyPayment * offer.loanTerm * 12) - offer.loanAmount).toFixed(2)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body1">
                    Total Cost:
                    <Tooltip title="The total amount you'll pay over the life of the loan, including principal and interest">
                      <InfoIcon fontSize="small" sx={{ ml: 1, verticalAlign: 'middle' }} />
                    </Tooltip>
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body1" align="right">
                    ${(offer.monthlyPayment * offer.loanTerm * 12).toFixed(2)}
                  </Typography>
                </Grid>
              </Grid>
              <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between' }}>
                <Button
                  variant="outlined"
                  color="primary"
                  startIcon={<RestartAltIcon />}
                  onClick={handleResetOffer}
                  disabled={!isOfferChanged}
                >
                  Reset
                </Button>
                <Button
                  variant="outlined"
                  color="primary"
                  startIcon={<CompareArrowsIcon />}
                  onClick={handleCompareOffers}
                >
                  Compare Offers
                </Button>
                <Button
                  variant="outlined"
                  color="primary"
                  startIcon={<BookmarkIcon />}
                  onClick={handleSaveForLater}
                >
                  Save for Later
                </Button>
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  onClick={handleAcceptOffer}
                  aria-label="Accept offer"
                >
                  Accept Offer
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      <Dialog
        open={openConfirmDialog}
        onClose={() => setOpenConfirmDialog(false)}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">{"Confirm Loan Offer Acceptance"}</DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            Are you sure you want to accept this loan offer? By accepting, you agree to the terms and conditions of the loan.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenConfirmDialog(false)} color="primary">
            Cancel
          </Button>
          <Button onClick={confirmAcceptOffer} color="primary" autoFocus disabled={isLoading}>
            {isLoading ? 'Processing...' : 'Confirm'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

export default OfferPage;