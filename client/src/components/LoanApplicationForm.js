import React, { useState } from 'react';
import { TextField, Button, Grid, MenuItem, CircularProgress } from '@mui/material';

const LoanApplicationForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    loan_amount: '',
    employment_length: '',
    employment_status: '',
    annual_income: '',
    debt_to_income_ratio: '',
    credit_score: '',
  });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);

  const validateForm = () => {
    const newErrors = {};
    if (!formData.loan_amount || formData.loan_amount <= 0) {
      newErrors.loan_amount = 'Loan amount must be greater than 0';
    }
    if (!formData.employment_length || formData.employment_length < 0) {
      newErrors.employment_length = 'Employment length must be 0 or greater';
    }
    if (!formData.employment_status) {
      newErrors.employment_status = 'Employment status is required';
    }
    if (!formData.annual_income || formData.annual_income <= 0) {
      newErrors.annual_income = 'Annual income must be greater than 0';
    }
    if (!formData.debt_to_income_ratio || formData.debt_to_income_ratio < 0 || formData.debt_to_income_ratio > 100) {
      newErrors.debt_to_income_ratio = 'Debt to income ratio must be between 0 and 100';
    }
    if (!formData.credit_score || formData.credit_score < 300 || formData.credit_score > 850) {
      newErrors.credit_score = 'Credit score must be between 300 and 850';
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    // Clear error when user starts typing
    if (errors[e.target.name]) {
      setErrors({ ...errors, [e.target.name]: null });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (validateForm()) {
      setLoading(true);
      await onSubmit(formData);
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <TextField
            fullWidth
            name="loan_amount"
            label="Loan Amount"
            type="number"
            value={formData.loan_amount}
            onChange={handleChange}
            required
            error={!!errors.loan_amount}
            helperText={errors.loan_amount}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            name="employment_length"
            label="Employment Length (years)"
            type="number"
            value={formData.employment_length}
            onChange={handleChange}
            required
            error={!!errors.employment_length}
            helperText={errors.employment_length}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            name="employment_status"
            label="Employment Status"
            select
            value={formData.employment_status}
            onChange={handleChange}
            required
            error={!!errors.employment_status}
            helperText={errors.employment_status}
          >
            <MenuItem value="full-time">Full-time</MenuItem>
            <MenuItem value="part-time">Part-time</MenuItem>
            <MenuItem value="self-employed">Self-employed</MenuItem>
            <MenuItem value="unemployed">Unemployed</MenuItem>
          </TextField>
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            name="annual_income"
            label="Annual Income"
            type="number"
            value={formData.annual_income}
            onChange={handleChange}
            required
            error={!!errors.annual_income}
            helperText={errors.annual_income}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            name="debt_to_income_ratio"
            label="Debt to Income Ratio"
            type="number"
            value={formData.debt_to_income_ratio}
            onChange={handleChange}
            required
            error={!!errors.debt_to_income_ratio}
            helperText={errors.debt_to_income_ratio}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            fullWidth
            name="credit_score"
            label="Credit Score"
            type="number"
            value={formData.credit_score}
            onChange={handleChange}
            required
            error={!!errors.credit_score}
            helperText={errors.credit_score}
          />
        </Grid>
        <Grid item xs={12}>
          <Button
            type="submit"
            variant="contained"
            color="primary"
            fullWidth
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Submit Application'}
          </Button>
        </Grid>
      </Grid>
    </form>
  );
};

export default LoanApplicationForm;