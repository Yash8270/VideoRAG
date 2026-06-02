import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const videoApi = {
  analyze: async (youtubeUrl, instagramUrl) => {
    const response = await api.post('/analyze', {
      youtube_url: youtubeUrl,
      instagram_url: instagramUrl,
    });
    return response.data;
  },
};

export const chatApi = {
  query: async (message, sessionId = null) => {
    const response = await api.post('/chat', {
      message,
      session_id: sessionId,
    });
    return response.data;
  },
};

export default api;
