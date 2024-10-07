import React, { useState } from 'react';
import { Form, Button, Alert, Container, Row, Col } from 'react-bootstrap';
import { API } from 'aws-amplify';

const LoanApplicationForm = () => {
  const [formData, setFormData] = useState({
    loanType: '',
    requestedAmount: '',
    creditScore: '',
    annualIncome: '',
    employmentStatus: '',
    homeOwnership: '',
    monthlyDebt: '',
    purpose: '',
    loanTerm: ''
  });

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [loanDecision, setLoanDecision] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    setErrors({ ...errors, [name]: '' });
  };

  const validateForm = () => {
    const newErrors = {};
    if (!formData.loanType) newErrors.loanType = 'Loan type is required';
    if (!formData.requestedAmount || isNaN(formData.requestedAmount) || formData.requestedAmount <= 0) {
      newErrors.requestedAmount = 'Please enter a valid amount';
    }
    if (!formData.creditScore || isNaN(formData.creditScore) || formData.creditScore < 300 || formData.creditScore > 850) {
      newErrors.creditScore = 'Please enter a valid credit score (300-850)';
    }
    if (!formData.annualIncome || isNaN(formData.annualIncome) || formData.annualIncome <= 0) {
      newErrors.annualIncome = 'Please enter a valid annual income';
    }
    if (!formData.employmentStatus) newErrors.employmentStatus = 'Employment status is required';
    if (!formData.homeOwnership) newErrors.homeOwnership = 'Home ownership status is required';
    if (!formData.monthlyDebt || isNaN(formData.monthlyDebt) || formData.monthlyDebt < 0) {
      newErrors.monthlyDebt = 'Please enter a valid monthly debt amount';
    }
    if (!formData.purpose) newErrors.purpose = 'Loan purpose is required';
    if (!formData.loanTerm) newErrors.loanTerm = 'Loan term is required';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (validateForm()) {
      setIsSubmitting(true);
      setSubmitError(null);
      try {
        const response = await API.post('loanAPI', '/application', { body: formData });
        console.log('Application submitted:', response);
        setSubmitSuccess(true);
        setLoanDecision(response);
      } catch (error) {
        console.error('Error submitting application:', error);
        setSubmitError('An error occurred while submitting your application. Please try again.');
      } finally {
        setIsSubmitting(false);
      }
    }
  };

  if (submitSuccess && loanDecision) {
    return (
      <Container>
        <Alert variant={loanDecision.approval_status ? "success" : "warning"}>
          <h4>{loanDecision.approval_status ? "Congratulations! Your loan has been approved." : "We're sorry, your loan application was not approved at this time."}</h4>
          <p>{loanDecision.llm_response.explanation}</p>
          {loanDecision.approval_status ? (
            <>
              <p>Approved Amount: ${loanDecision.llm_response.approved_amount}</p>
              <p>Interest Rate: {loanDecision.llm_response.interest_rate}%</p>
              <p>Loan Term: {loanDecision.llm_response.loan_term} months</p>
              <p>Monthly Payment: ${loanDecision.llm_response.monthly_payment}</p>
            </>
          ) : (
            <>
              <h5>Suggestions for Improvement:</h5>
              <ul>
                {loanDecision.llm_response.improvement_suggestions.map((suggestion, index) => (
                  <li key={index}>{suggestion}</li>
                ))}
              </ul>
            </>
          )}
        </Alert>
      </Container>
    );
  }

  return (
    <Container>
      <Form onSubmit={handleSubmit}>
        <Row>
          <Col md={6}>
            <Form.Group>
              <Form.Label>Loan Type</Form.Label>
              <Form.Control
                as="select"
                name="loanType"
                value={formData.loanType}
                onChange={handleChange}
                isInvalid={!!errors.loanType}
              >
                <option value="">Select loan type</option>
                <option value="personal">Personal Loan</option>
                <option value="auto">Auto Loan</option>
                <option value="mortgage">Mortgage</option>
                <option value="business">Business Loan</option>
              </Form.Control>
              <Form.Control.Feedback type="invalid">{errors.loanType}</Form.Control.Feedback>
            </Form.Group>
          </Col>
          <Col md={6}>
            <Form.Group>
              <Form.Label>Requested Amount ($)</Form.Label>
              <Form.Control
                type="number"
                name="requestedAmount"
                value={formData.requestedAmount}
                onChange={handleChange}
                isInvalid={!!errors.requestedAmount}
              />
              <Form.Control.Feedback type="invalid">{errors.requestedAmount}</Form.Control.Feedback>
            </Form.Group>
          </Col>
        </Row>

        {/* Add similar Row and Col structures for other form fields */}

        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Submitting...' : 'Submit Application'}
        </Button>

        {submitError && <Alert variant="danger" className="mt-3">{submitError}</Alert>}
      </Form>
    </Container>
  );
};

export default LoanApplicationForm;