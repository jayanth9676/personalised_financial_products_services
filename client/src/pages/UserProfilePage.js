import React, { useState, useEffect } from 'react';
import { Container, Typography, TextField, Button, Box, CircularProgress } from '@mui/material';
// import { API } from 'aws-amplify';
import { useSnackbar } from 'notistack';
import { Formik, Form, Field } from 'formik';
import * as Yup from 'yup';

const validationSchema = Yup.object().shape({
  fullName: Yup.string().required('Full name is required'),
  email: Yup.string().email('Invalid email').required('Email is required'),
  phoneNumber: Yup.string().matches(/^\d{10}$/, 'Phone number must be 10 digits'),
  address: Yup.string().required('Address is required'),
});

function UserProfilePage() {
  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const { enqueueSnackbar } = useSnackbar();

  useEffect(() => {
    fetchUserProfile();
  }, []);

  const fetchUserProfile = async () => {
    try {
      // Simulating API call
      const mockProfile = {
        fullName: 'John Doe',
        email: 'john.doe@example.com',
        phoneNumber: '1234567890',
        address: '123 Main St, Anytown, USA'
      };
      setUserProfile(mockProfile);
    } catch (error) {
      console.error('Error fetching user profile:', error);
      enqueueSnackbar('Failed to fetch user profile', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (values, { setSubmitting }) => {
    try {
      // Simulating API call
      console.log('Profile updated:', values);
      enqueueSnackbar('Profile updated successfully', { variant: 'success' });
    } catch (error) {
      console.error('Error updating profile:', error);
      enqueueSnackbar('Failed to update profile', { variant: 'error' });
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="md" sx={{ mt: 4, textAlign: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h3" gutterBottom align="center">
        User Profile
      </Typography>
      <Formik
        initialValues={userProfile}
        validationSchema={validationSchema}
        onSubmit={handleSubmit}
      >
        {({ errors, touched, isSubmitting }) => (
          <Form>
            <Field
              as={TextField}
              fullWidth
              label="Full Name"
              name="fullName"
              error={touched.fullName && errors.fullName}
              helperText={touched.fullName && errors.fullName}
              margin="normal"
              required
            />
            <Field
              as={TextField}
              fullWidth
              label="Email"
              name="email"
              type="email"
              error={touched.email && errors.email}
              helperText={touched.email && errors.email}
              margin="normal"
              required
            />
            <Field
              as={TextField}
              fullWidth
              label="Phone Number"
              name="phoneNumber"
              error={touched.phoneNumber && errors.phoneNumber}
              helperText={touched.phoneNumber && errors.phoneNumber}
              margin="normal"
            />
            <Field
              as={TextField}
              fullWidth
              label="Address"
              name="address"
              error={touched.address && errors.address}
              helperText={touched.address && errors.address}
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
                Update Profile
              </Button>
            </Box>
          </Form>
        )}
      </Formik>
    </Container>
  );
}

export default UserProfilePage;