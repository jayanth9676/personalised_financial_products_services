import React from 'react';
import { Box, Typography, Link } from '@mui/material';
import { styled } from '@mui/system';

const StyledFooter = styled(Box)`
  background: linear-gradient(45deg, #3a0ca3 30%, #4cc9f0 90%);
  color: white;
  padding: 20px;
  text-align: center;
`;

function Footer() {
  return (
    <StyledFooter>
      <Typography variant="body2">
        © 2024 FutureBank AI. All rights reserved.
      </Typography>
      <Box mt={1}>
        <Link href="#" color="inherit" mx={1}>
          Privacy Policy
        </Link>
        |
        <Link href="#" color="inherit" mx={1}>
          Terms of Service
        </Link>
      </Box>
    </StyledFooter>
  );
}

export default Footer;