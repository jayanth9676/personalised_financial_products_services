exports.handler = async (event) => {
  console.log('WebSocket Disconnect:', event);
  return { statusCode: 200, body: 'Disconnected' };
};