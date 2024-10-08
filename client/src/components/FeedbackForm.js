import React, { useState } from 'react';
import { API } from 'aws-amplify';

const FeedbackForm = ({ responseId }) => {
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await API.post('loanAPI', '/feedback', {
        body: {
          responseId,
          rating,
          comment,
          userId: 'current-user-id', // You'd get this from your auth system
        },
      });
      alert('Feedback submitted successfully!');
      setRating(0);
      setComment('');
    } catch (error) {
      console.error('Error submitting feedback:', error);
      alert('Error submitting feedback. Please try again.');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h3>How helpful was this response?</h3>
      {[1, 2, 3, 4, 5].map((value) => (
        <button
          key={value}
          type="button"
          onClick={() => setRating(value)}
          style={{ backgroundColor: rating >= value ? 'gold' : 'white' }}
        >
          ★
        </button>
      ))}
      <textarea
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        placeholder="Any additional comments?"
      />
      <button type="submit">Submit Feedback</button>
    </form>
  );
};

export default FeedbackForm;