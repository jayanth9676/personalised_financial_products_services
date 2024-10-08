import React from 'react';
import { Link } from 'react-router-dom';

const Navigation = () => {
  const isLoggedIn = !!localStorage.getItem('token');

  return (
    <nav>
      <ul>
        <li><Link to="/loan-application">Loan Application</Link></li>
        {isLoggedIn ? (
          <li><button onClick={() => {
            localStorage.removeItem('token');
            window.location.reload();
          }}>Logout</button></li>
        ) : (
          <li><Link to="/login">Login</Link></li>
        )}
      </ul>
    </nav>
  );
};

export default Navigation;