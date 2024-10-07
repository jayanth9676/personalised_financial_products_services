import React from 'react';
import { Stepper, Step, StepLabel } from '@mui/material';

const steps = ['Personal Information', 'Loan Details', 'Review Offer', 'Accept Offer'];

function LoanApplicationStepper({ activeStep }) {
  return (
    <Stepper activeStep={activeStep} alternativeLabel>
      {steps.map((label) => (
        <Step key={label}>
          <StepLabel>{label}</StepLabel>
        </Step>
      ))}
    </Stepper>
  );
}

export default LoanApplicationStepper;