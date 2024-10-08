import React, { useState } from 'react';
import { API } from 'aws-amplify';
import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';

const schema = yup.object().shape({
  loanAmount: yup.number().positive().required('Loan amount is required'),
  loanPurpose: yup.string().required('Loan purpose is required'),
  annualIncome: yup.number().positive().required('Annual income is required'),
  employmentStatus: yup.string().required('Employment status is required'),
  creditScore: yup.number().positive().integer().required('Credit score is required'),
});

const LoanApplicationPage = () => {
  const [step, setStep] = useState(1);
  const { register, handleSubmit, formState: { errors } } = useForm({
    resolver: yupResolver(schema),
  });

  const onSubmit = async (data) => {
    try {
      const response = await API.post('loanAPI', '/apply', { body: data });
      // Handle successful submission (e.g., redirect to status page)
    } catch (error) {
      console.error('Error submitting loan application:', error);
      // Handle error (e.g., show error message to user)
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      {step === 1 && (
        <>
          <input {...register('loanAmount')} placeholder="Loan Amount" />
          <input {...register('loanPurpose')} placeholder="Loan Purpose" />
          <button type="button" onClick={() => setStep(2)}>Next</button>
        </>
      )}
      {step === 2 && (
        <>
          <input {...register('annualIncome')} placeholder="Annual Income" />
          <input {...register('employmentStatus')} placeholder="Employment Status" />
          <input {...register('creditScore')} placeholder="Credit Score" />
          <button type="submit">Submit Application</button>
        </>
      )}
      {errors && (
        <div className="errors">
          {Object.values(errors).map((error, index) => (
            <p key={index}>{error.message}</p>
          ))}
        </div>
      )}
    </form>
  );
};

export default LoanApplicationPage;