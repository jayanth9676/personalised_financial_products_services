import React, { useState, useEffect, useRef } from 'react';
import { Container, Typography, Paper, TextField, Button, List, ListItem, ListItemText, CircularProgress, Alert } from '@mui/material';
import { useWebSocket } from '../components/WebSocketProvider';

function AIAssistantPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const webSocket = useWebSocket();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
    webSocket.subscribe('ai_response', handleAIResponse);
    webSocket.subscribe('error', handleError);
    return () => {
      webSocket.unsubscribe('ai_response', handleAIResponse);
      webSocket.unsubscribe('error', handleError);
    };
  }, [messages, webSocket]);

  const handleAIResponse = (response) => {
    setMessages(prev => [...prev, { text: response.message, sender: 'ai' }]);
    setIsLoading(false);
  };

  const handleError = (errorMessage) => {
    setError(errorMessage);
    setIsLoading(false);
  };

  const handleSend = async () => {
    if (input.trim() === '') return;

    setIsLoading(true);
    setMessages(prev => [...prev, { text: input, sender: 'user' }]);
    setInput('');

    webSocket.send('user_message', { message: input });
  };

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mt: 4, height: '70vh', display: 'flex', flexDirection: 'column' }}>
        <Typography variant="h4" gutterBottom>AI Assistant</Typography>
        <List sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
          {messages.map((message, index) => (
            <ListItem key={index} sx={{ justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start' }}>
              <Paper elevation={1} sx={{ p: 2, bgcolor: message.sender === 'user' ? 'primary.light' : 'secondary.light' }}>
                <ListItemText primary={message.text} />
              </Paper>
            </ListItem>
          ))}
          <div ref={messagesEndRef} />
        </List>
        <TextField
          fullWidth
          variant="outlined"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          disabled={isLoading}
        />
        <Button
          variant="contained"
          onClick={handleSend}
          disabled={isLoading}
          sx={{ mt: 2 }}
        >
          {isLoading ? <CircularProgress size={24} /> : 'Send'}
        </Button>
        {error && <Alert severity="error" onClose={() => setError(null)}>{error}</Alert>}
      </Paper>
    </Container>
  );
}

export default AIAssistantPage;