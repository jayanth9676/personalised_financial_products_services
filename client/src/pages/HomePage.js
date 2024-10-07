import React from 'react';
import { Container, Typography, Button, Box, Grid, Card, CardContent } from '@mui/material';
import { styled } from '@mui/system';
import { Link } from 'react-router-dom';

const HeroSection = styled(Box)`
  background: linear-gradient(45deg, #3a0ca3 30%, #4cc9f0 90%);
  color: white;
  padding: 100px 0;
  text-align: center;
`;

const FeatureCard = styled(Card)`
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
`;

function HomePage() {
  return (
    <div>
      <HeroSection>
        <Container>
          <Typography variant="h2" gutterBottom>
            Welcome to FutureBank AI
          </Typography>
          <Typography variant="h5" paragraph>
            Experience the future of personalized banking with AI-powered recommendations
          </Typography>
          <Button
            component={Link}
            to="/apply"
            variant="contained"
            color="secondary"
            size="large"
          >
            Apply for a Loan
          </Button>
        </Container>
      </HeroSection>
      <Container sx={{ my: 8 }}>
        <Typography variant="h3" gutterBottom align="center">
          Our AI-Powered Features
        </Typography>
        <Grid container spacing={4} sx={{ mt: 4 }}>
          <Grid item xs={12} md={4}>
            <FeatureCard>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Personalized Loan Recommendations
                </Typography>
                <Typography variant="body1">
                  Our AI analyzes your financial profile to offer tailored loan options that fit your needs.
                </Typography>
              </CardContent>
            </FeatureCard>
          </Grid>
          <Grid item xs={12} md={4}>
            <FeatureCard>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  Real-Time Offer Adjustments
                </Typography>
                <Typography variant="body1">
                  Instantly see how changing loan parameters affects your offers, powered by advanced AI.
                </Typography>
              </CardContent>
            </FeatureCard>
          </Grid>
          <Grid item xs={12} md={4}>
            <FeatureCard>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  AI-Assisted Application Process
                </Typography>
                <Typography variant="body1">
                  Our intelligent system guides you through the application, making it quick and easy.
                </Typography>
              </CardContent>
            </FeatureCard>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
}

export default HomePage;