import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { ErrorBoundary } from 'react-error-boundary';
import { SnackbarProvider } from 'notistack';
import Header from './components/Header';
import Footer from './components/Footer';
import LoadingFallback from './components/LoadingFallback';
import ErrorFallback from './components/ErrorFallback';
import './App.css';

// Lazy load pages for better performance
const HomePage = lazy(() => import('./pages/HomePage'));
const LoanRecommendationsPage = lazy(() => import('./pages/LoanRecommendationsPage'));
const LoanApplicationPage = lazy(() => import('./pages/LoanApplicationPage'));
const OfferPage = lazy(() => import('./pages/OfferPage'));
const UserProfilePage = lazy(() => import('./pages/UserProfilePage'));
const AIAssistantPage = lazy(() => import('./pages/AIAssistantPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const CompareOffersPage = lazy(() => import('./pages/CompareOffersPage'));

const theme = createTheme({
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

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SnackbarProvider maxSnack={3}>
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <div className="App">
            <Header />
            <main>
              <Suspense fallback={<LoadingFallback />}>
                <Routes>
                  <Route path="/" element={<HomePage />} />
                  <Route path="/recommendations" element={<LoanRecommendationsPage />} />
                  <Route path="/apply" element={<Navigate to="/recommendations" replace />} />
                  <Route path="/apply/:loanType" element={<LoanApplicationPage />} />
                  <Route path="/offer" element={<OfferPage />} />
                  <Route path="/profile" element={<UserProfilePage />} />
                  <Route path="/ai-assistant" element={<AIAssistantPage />} />
                  <Route path="/dashboard" element={<DashboardPage />} />
                  <Route path="/compare-offers" element={<CompareOffersPage />} />
                </Routes>
              </Suspense>
            </main>
            <Footer />
          </div>
        </ErrorBoundary>
      </SnackbarProvider>
    </ThemeProvider>
  );
}

export default App;
