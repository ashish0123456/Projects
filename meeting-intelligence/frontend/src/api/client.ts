import axios from 'axios';

const client = axios.create({
  baseURL: '/api/v1', // Proxy to backend server
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
});

export default client;