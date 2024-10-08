import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter } from 'react-router-dom';
import { ABTestProvider } from './components/ABTestProvider';
import App from './App';

ReactDOM.render(
  <React.StrictMode>
    <BrowserRouter>
      <ABTestProvider>
        <App />
      </ABTestProvider>
    </BrowserRouter>
  </React.StrictMode>,
  document.getElementById('root')
);