import React from 'react';
import { styled } from '@mui/material/styles';

const StyledSkipLink = styled('a')(({ theme }) => ({
  position: 'absolute',
  top: '-40px',
  left: 0,
  background: theme.palette.primary.main,
  color: theme.palette.primary.contrastText,
  padding: theme.spacing(1),
  zIndex: 100,
  '&:focus': {
    top: 0,
  },
}));

function SkipLink() {
  return (
    <StyledSkipLink href="#main-content">
      Skip to main content
    </StyledSkipLink>
  );
}

export default SkipLink;