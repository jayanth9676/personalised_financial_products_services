import React from 'react';
import { Typography, Button, Box } from '@mui/material';

function ErrorFallback({ error, resetErrorBoundary }) {
  return (
    <Box
      display="flex"
      flexDirection="column"
      justifyContent="center"
      alignItems="center"
      height="100vh"
    >
      <Typography variant="h5" gutterBottom>
        Oops! Something went wrong.
      </Typography>
      <Typography variant="body1" gutterBottom>
        {error.message}
      </Typography>
      <Button variant="contained" color="primary" onClick={resetErrorBoundary}>
        Try again
      </Button>
    </Box>
  );
}

export default ErrorFallback;