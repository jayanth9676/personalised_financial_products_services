import { API } from 'aws-amplify';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.listeners = new Map();
  }

  connect() {
    return API.get('LoanAPI', '/websocket-url')
      .then(url => {
        this.socket = new WebSocket(url);
        this.socket.onmessage = this.handleMessage.bind(this);
        return new Promise((resolve, reject) => {
          this.socket.onopen = resolve;
          this.socket.onerror = reject;
        });
      });
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
    }
  }

  subscribe(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
  }

  unsubscribe(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }

  handleMessage(event) {
    const data = JSON.parse(event.data);
    if (this.listeners.has(data.event)) {
      this.listeners.get(data.event).forEach(callback => callback(data.payload));
    }
  }

  send(event, payload) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ event, payload }));
    }
  }
}

export default new WebSocketService();