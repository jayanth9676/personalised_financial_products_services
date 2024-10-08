import React, { useState, useEffect } from 'react';
import axios from 'axios';

const LoanAdjuster = ({ initialLoanData, onAdjustment }) => {
  const [loanData, setLoanData] = useState(initialLoanData);
  const [adjustedLoan, setAdjustedLoan] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoanData(initialLoanData);
  }, [initialLoanData]);

  const handleInputChange = (e) => {
    setLoanData({ ...loanData, [e.target.name]: parseFloat(e.target.value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      const response = await axios.post('/api/adjust-loan-parameters', loanData);
      setAdjustedLoan(response.data);
      if (onAdjustment) onAdjustment(response.data);
    } catch (error) {
      console.error('Error adjusting loan parameters:', error);
      setError('Failed to adjust loan parameters. Please try again.');
    }
  };

  return (
    <div className="loan-adjuster">
      <h2>Adjust Loan Parameters</h2>
      <form onSubmit={handleSubmit}>
        {/* Add more input fields for all required parameters */}
        <input
          type="number"
          name="loan_amount"
          value={loanData.loan_amount}
          onChange={handleInputChange}
          placeholder="Loan Amount"
        />
        <input
          type="number"
          name="loan_term"
          value={loanData.loan_term}
          onChange={handleInputChange}
          placeholder="Loan Term (months)"
        />
        <input
          type="number"
          name="credit_score"
          value={loanData.credit_score}
          onChange={handleInputChange}
          placeholder="Credit Score"
        />
        <input
          type="number"
          name="income"
          value={loanData.income}
          onChange={handleInputChange}
          placeholder="Annual Income"
        />
        <input
          type="number"
          name="debt_to_income_ratio"
          value={loanData.debt_to_income_ratio}
          onChange={handleInputChange}
          placeholder="Debt-to-Income Ratio"
          step="0.01"
        />
        <button type="submit">Adjust Loan</button>
      </form>
      {error && <p className="error">{error}</p>}
      {adjustedLoan && (
        <div className="adjusted-loan-details">
          <h3>Adjusted Loan Details</h3>
          <p>Interest Rate: {adjustedLoan.interest_rate.toFixed(2)}%</p>
          <p>Monthly Payment: ${adjustedLoan.monthly_payment.toFixed(2)}</p>
          <p>Approval Probability: {(adjustedLoan.approval_probability * 100).toFixed(2)}%</p>
          <h4>Explanation</h4>
          <p>{adjustedLoan.explanation}</p>
        </div>
      )}
    </div>
  );
};

export default LoanAdjuster;