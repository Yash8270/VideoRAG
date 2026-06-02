import React, { useState } from 'react';
import VideoContext from './VideoContext';

const Api = (props) => {
  // Use VITE_API_URL from environment when deployed, fallback to localhost for development
  const host = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

  // Global Application State
  const [analysisData, setAnalysisData] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // ── analyzeVideos ─────────────────────────────────────────────
  const analyzeVideos = async (youtubeUrl, instagramUrl) => {
    setIsLoading(true);
    setError('');
    setAnalysisData(null);
    try {
      const response = await fetch(`${host}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          youtube_url: youtubeUrl,
          instagram_url: instagramUrl,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to analyze videos');
      }

      const data = await response.json();
      setAnalysisData(data);
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // ── chatWithVideos ────────────────────────────────────────────
  const chatWithVideos = async (message, currentSessionId, videoIds) => {
    try {
      const response = await fetch(`${host}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          session_id: currentSessionId,
          video_ids: videoIds,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to connect to RAG backend');
      }

      const data = await response.json();
      
      // Save session globally if it's the first message
      if (data.session_id && !currentSessionId) {
        setSessionId(data.session_id);
      }
      
      return data;
    } catch (err) {
      throw err;
    }
  };

  return (
    <VideoContext.Provider value={{
      analysisData, setAnalysisData,
      sessionId, setSessionId,
      isLoading, setIsLoading,
      error, setError,
      analyzeVideos,
      chatWithVideos
    }}>
      {props.children}
    </VideoContext.Provider>
  );
};

export default Api;
