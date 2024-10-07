import React, { useState, useEffect } from 'react';
import { Container, Typography, TextField, Button, Box, Card, CardContent } from '@mui/material';
import { API } from 'aws-amplify';

function ChatbotPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const userMessage = { text: input, sender: 'user' };
    setMessages([...messages, userMessage]);
    setInput('');

    try {
      const response = await API.post('chatbotAPI', '/chat', { body: { message: input } });
      const botMessage = { text: response.message, sender: 'bot' };
      setMessages(prevMessages => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      // Show error message to the user
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h3" gutterBottom align="center">
        AI Loan Assistant
      </Typography>
      <Card>
        <CardContent>
          <Box sx={{ height: 400, overflowY: 'auto', mb: 2 }}>
            {messages.map((message, index) => (
              <Typography key={index} align={message.sender === 'user' ? 'right' : 'left'}>
                {message.text}
              </Typography>
            ))}
          </Box>
          <Box sx={{ display: 'flex' }}>
            <TextField
              fullWidth
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about loans..."
            />
            <Button variant="contained" onClick={sendMessage} sx={{ ml: 1 }}>
              Send
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Container>
  );
}

export default ChatbotPage;