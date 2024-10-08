import React from 'react';
import { BrowserRouter as Router, Route, Switch, Redirect } from 'react-router-dom';
import LoanApplication from './components/LoanApplication';
import Login from './components/Login';
import PrivateRoute from './components/PrivateRoute';
import Navigation from './components/Navigation';

function App() {
  return (
    <Router>
      <div className="App">
        <Navigation />
        <Switch>
          <Route exact path="/login" component={Login} />
          <PrivateRoute exact path="/loan-application" component={LoanApplication} />
          <Redirect from="/" to="/loan-application" />
        </Switch>
      </div>
    </Router>
  );
}

export default App;