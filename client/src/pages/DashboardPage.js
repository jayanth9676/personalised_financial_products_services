import React from 'react';
import { Container, Typography, Grid, Card, CardContent, Box } from '@mui/material';
import { styled } from '@mui/system';

const StyledCard = styled(Card)`
  height: 100%;
`;

function DashboardPage() {
  // This data would typically come from your backend
  const userData = {
    name: 'John Doe',
    creditScore: 750,
    loanApplications: [
      { id: 1, type: 'Home Loan', amount: 250000, status: 'Approved' },
      { id: 2, type: 'Car Loan', amount: 35000, status: 'Pending' },
    ],
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Typography variant="h3" gutterBottom>
        Welcome, {userData.name}
      </Typography>
      <Grid container spacing={4}>
        <Grid item xs={12} md={4}>
          <StyledCard>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Credit Score
              </Typography>
              <Typography variant="h3">{userData.creditScore}</Typography>
            </CardContent>
          </StyledCard>
        </Grid>
        <Grid item xs={12} md={8}>
          <StyledCard>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Loan Applications
              </Typography>
              {userData.loanApplications.map((loan) => (
                <Box key={loan.id} sx={{ mb: 2 }}>
                  <Typography variant="subtitle1">
                    {loan.type} - ${loan.amount.toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Status: {loan.status}
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </StyledCard>
        </Grid>
      </Grid>
    </Container>
  );
}

export default DashboardPage;