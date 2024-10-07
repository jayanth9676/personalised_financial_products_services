import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link } from 'react-router-dom';
import { styled } from '@mui/system';

const StyledAppBar = styled(AppBar)`
  background: linear-gradient(45deg, #3a0ca3 30%, #4cc9f0 90%);
`;

const StyledLink = styled(Link)`
  color: white;
  text-decoration: none;
  margin: 0 10px;
`;

function Header() {
  return (
    <StyledAppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          FutureBank AI
        </Typography>
        <Box>
          <StyledLink to="/">
            <Button color="inherit">Home</Button>
          </StyledLink>
          <StyledLink to="/recommendations">
            <Button color="inherit">Apply for Loan</Button>
          </StyledLink>
          <StyledLink to="/dashboard">
            <Button color="inherit">Dashboard</Button>
          </StyledLink>
        </Box>
      </Toolbar>
    </StyledAppBar>
  );
}

export default Header;