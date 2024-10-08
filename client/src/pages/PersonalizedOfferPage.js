import React from 'react';
import FeedbackForm from '../components/FeedbackForm';

const PersonalizedOfferPage = () => {
  // Assume we have an 'offer' object with offer details
  const offer = { id: 'offer-123', /* other offer details */ };

  return (
    <div>
      <h2>Your Personalized Loan Offer</h2>
      {/* Display offer details */}
      <FeedbackForm responseId={offer.id} />
    </div>
  );
};

export default PersonalizedOfferPage;