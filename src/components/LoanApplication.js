import React, { useState, useEffect } from 'react';
import axios from 'axios';
import LoanAdjuster from './LoanAdjuster';

const LoanApplication = () => {
  const [userData, setUserData] = useState(null);
  const [personalizedLoans, setPersonalizedLoans] = useState([]);
  const [selectedLoan, setSelectedLoan] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [userDataResponse, loansResponse] = await Promise.all([
        axios.get('/api/user-data'),
        axios.get('/api/personalized-loans')
      ]);
      setUserData(userDataResponse.data);
      setPersonalizedLoans(loansResponse.data);
    } catch (error) {
      console.error('Error fetching data:', error);
      setError('Failed to fetch data. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchPersonalizedLoans = async () => {
    try {
      const response = await axios.get('/api/personalized-loans');
      setPersonalizedLoans(response.data);
    } catch (error) {
      console.error('Error fetching personalized loans:', error);
      setError('Failed to fetch personalized loans. Please try again.');
    }
  };

  const handleLoanSelection = (loan) => {
    setSelectedLoan(loan);
  };

  const handleLoanAdjustment = (adjustedLoan) => {
    setSelectedLoan(adjustedLoan);
  };

  const handleAcceptOffer = async () => {
    try {
      await axios.post('/api/accept-offer', selectedLoan);
      alert('Offer accepted successfully!');
      // Optionally, redirect to a confirmation page or update UI
    } catch (error) {
      console.error('Error accepting offer:', error);
      setError('Failed to accept offer. Please try again.');
    }
  };

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="loan-application">
      <h1>Loan Application</h1>
      {userData && (
        <div className="user-info">
          <h2>User Information</h2>
          <p>Credit Score: {userData.credit_score}</p>
          <p>Annual Income: ${userData.income}</p>
          <p>Debt-to-Income Ratio: {userData.debt_to_income_ratio}</p>
        </div>
      )}
      <div className="personalized-loans">
        <h2>Personalized Loan Options</h2>
        {personalizedLoans.map((loan, index) => (
          <div key={index} className="loan-option" onClick={() => handleLoanSelection(loan)}>
            <h3>{loan.loan_type}</h3>
            <p>Max Amount: ${loan.max_amount}</p>
            <p>Interest Rate: {loan.interest_rate}%</p>
            <p>Loan Term: {loan.loan_term} months</p>
            <p>{loan.description}</p>
          </div>
        ))}
      </div>
      {selectedLoan && (
        <div className="selected-loan">
          <h2>Selected Loan</h2>
          <LoanAdjuster initialLoanData={selectedLoan} onAdjustment={handleLoanAdjustment} />
          <button onClick={handleAcceptOffer}>Accept Offer</button>
        </div>
      )}
    </div>
  );
};

export default LoanApplication;