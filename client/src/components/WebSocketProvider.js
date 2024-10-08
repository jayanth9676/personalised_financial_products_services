import React, { createContext, useContext, useEffect } from 'react';
import WebSocketService from '../services/websocketService';

const WebSocketContext = createContext();

export const useWebSocket = () => useContext(WebSocketContext);

function WebSocketProvider({ children }) {
  useEffect(() => {
    WebSocketService.connect();
    return () => WebSocketService.disconnect();
  }, []);

  return (
    <WebSocketContext.Provider value={WebSocketService}>
      {children}
    </WebSocketContext.Provider>
  );
}

export default WebSocketProvider;