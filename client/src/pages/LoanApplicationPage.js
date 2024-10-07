import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Typography, TextField, Button, Box, Card } from '@mui/material';
import { styled } from '@mui/system';
// import { API } from 'aws-amplify';
import { Formik, Form, Field } from 'formik';
import * as Yup from 'yup';
import { useSnackbar } from 'notistack';

const StyledCard = styled(Card)`
  padding: 24px;
  margin-top: 24px;
`;

const validationSchema = Yup.object().shape({
  loanAmount: Yup.number()
    .min(1000, 'Loan amount must be at least $1,000')
    .max(1000000, 'Loan amount cannot exceed $1,000,000')
    .required('Loan amount is required'),
});

function LoanApplicationPage() {
  const { loanType } = useParams();
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();

  const initialValues = {
    loanType: loanType,
    loanAmount: '',
    // Add more fields as needed
  };

  useEffect(() => {
    // Commented out fetchUserData as it requires AWS
    // fetchUserData();
  }, []);

  // const fetchUserData = async () => {
  //   try {
  //     const userData = await API.get('loanAPI', '/user');
  //     // Update initialValues with user data if needed
  //   } catch (error) {
  //     console.error('Error fetching user data:', error);
  //     enqueueSnackbar('Failed to fetch user data', { variant: 'error' });
  //   }
  // };

  const handleSubmit = async (values, { setSubmitting }) => {
    try {
      // Simulating API call
      console.log('Application submitted:', values);
      enqueueSnackbar('Loan application submitted successfully', { variant: 'success' });
      navigate('/offer', { state: { offer: {
        loanAmount: values.loanAmount,
        interestRate: 5.5,
        loanTerm: 5,
        monthlyPayment: (values.loanAmount * 0.055) / 12,
        minAmount: 1000,
        maxAmount: 100000,
        minTerm: 1,
        maxTerm: 10
      } } });
    } catch (error) {
      console.error('Error submitting application:', error);
      enqueueSnackbar('Failed to submit loan application', { variant: 'error' });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Typography variant="h3" gutterBottom align="center" sx={{ mt: 4 }}>
        Apply for {loanType} Loan
      </Typography>
      <StyledCard>
        <Formik
          initialValues={initialValues}
          validationSchema={validationSchema}
          onSubmit={handleSubmit}
        >
          {({ errors, touched, isSubmitting }) => (
            <Form>
              <Field
                as={TextField}
                fullWidth
                label="Loan Amount"
                name="loanAmount"
                type="number"
                error={touched.loanAmount && errors.loanAmount}
                helperText={touched.loanAmount && errors.loanAmount}
                margin="normal"
                required
              />
              <Box sx={{ mt: 3, textAlign: 'center' }}>
                <Button
                  variant="contained"
                  color="primary"
                  type="submit"
                  size="large"
                  disabled={isSubmitting}
                >
                  Get Personalized Offer
                </Button>
              </Box>
            </Form>
          )}
        </Formik>
      </StyledCard>
    </Container>
  );
}

export default LoanApplicationPage;