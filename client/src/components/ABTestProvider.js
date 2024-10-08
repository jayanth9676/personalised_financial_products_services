import React, { createContext, useState, useEffect } from 'react';
import { API } from 'aws-amplify';

export const ABTestContext = createContext();

export function ABTestProvider({ children }) {
  const [variant, setVariant] = useState(null);

  useEffect(() => {
    async function fetchVariant() {
      try {
        const response = await API.get('LoanAPI', '/ab-test-variant');
        setVariant(response.variant);
      } catch (error) {
        console.error('Error fetching A/B test variant:', error);
        setVariant('A'); // Default to variant A if there's an error
      }
    }
    fetchVariant();
  }, []);

  return (
    <ABTestContext.Provider value={variant}>
      {children}
    </ABTestContext.Provider>
  );
}