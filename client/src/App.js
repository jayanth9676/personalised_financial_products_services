import React, { Suspense, lazy, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, responsiveFontSizes } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { ErrorBoundary } from 'react-error-boundary';
import { SnackbarProvider } from 'notistack';
import { Amplify, Analytics } from 'aws-amplify';
import { withAuthenticator } from '@aws-amplify/ui-react';
import Header from './components/Header';
import Footer from './components/Footer';
import LoadingFallback from './components/LoadingFallback';
import ErrorFallback from './components/ErrorFallback';
import SkipLink from './components/SkipLink';
import WebSocketProvider from './components/WebSocketProvider';
import awsConfig from './aws-exports';
import './App.css';
import { ABTestProvider } from './components/ABTestProvider';
import FeedbackForm from './components/FeedbackForm';

Amplify.configure(awsConfig);

// Lazy load pages for better performance
const HomePage = lazy(() => import('./pages/HomePage'));
const LoanRecommendationsPage = lazy(() => import('./pages/LoanRecommendationsPage'));
const LoanApplicationPage = lazy(() => import('./pages/LoanApplicationPage'));
const PersonalizedOfferPage = lazy(() => import('./pages/PersonalizedOfferPage'));
const CompareOffersPage = lazy(() => import('./pages/CompareOffersPage'));
const AIAssistantPage = lazy(() => import('./pages/AIAssistantPage'));
const UserProfilePage = lazy(() => import('./pages/UserProfilePage'));

// Make the theme responsive
let theme = createTheme({
  palette: {
    primary: {
      main: '#3a0ca3',
    },
    secondary: {
      main: '#4cc9f0',
    },
    background: {
      default: '#f0f0f0',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
    },
    h2: {
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 30,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
        },
      },
    },
  },
});
theme = responsiveFontSizes(theme);

function App() {
  useEffect(() => {
    // Set up CloudWatch logging
    Analytics.autoTrack('session', {
      enable: true,
      provider: 'AWSPinpoint'
    });
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SnackbarProvider maxSnack={3}>
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <ABTestProvider>
            <WebSocketProvider>
              <div className="App">
                <SkipLink />
                <Header />
                <main id="main-content">
                  <Suspense fallback={<LoadingFallback />}>
                    <Routes>
                      <Route path="/" element={<HomePage />} />
                      <Route path="/recommendations" element={<LoanRecommendationsPage />} />
                      <Route path="/apply/:loanType" element={<LoanApplicationPage />} />
                      <Route path="/personalized-offer" element={<PersonalizedOfferPage />} />
                      <Route path="/compare-offers" element={<CompareOffersPage />} />
                      <Route path="/ai-assistant" element={<AIAssistantPage />} />
                      <Route path="/profile" element={<UserProfilePage />} />
                      <Route path="*" element={<Navigate to="/" replace />} />
                    </Routes>
                  </Suspense>
                </main>
                <Footer />
                <FeedbackForm />
              </div>
            </WebSocketProvider>
          </ABTestProvider>
        </ErrorBoundary>
      </SnackbarProvider>
    </ThemeProvider>
  );
}

export default withAuthenticator(App);
