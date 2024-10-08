exports.handler = async (event) => {
  console.log('WebSocket Connect:', event);
  return { statusCode: 200, body: 'Connected' };
};