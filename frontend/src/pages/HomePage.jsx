import React, { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import VideoContext from '../context/VideoContext';
import { Youtube, Instagram, ArrowRight, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

const HomePage = () => {
  const [ytUrl, setYtUrl] = useState('');
  const [igUrl, setIgUrl] = useState('');
  const navigate = useNavigate();
  
  const { isLoading, error, setError, analyzeVideos } = useContext(VideoContext);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!ytUrl || !igUrl) {
      if (setError) setError("Please provide both URLs.");
      return;
    }
    
    try {
      await analyzeVideos(ytUrl, igUrl);
      // On success, navigate to the results page
      navigate('/results');
    } catch (err) {
      // Error is handled and shown by the context
    }
  };

  if (isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center space-y-6 min-h-[calc(100vh-88px)]">
        <div className="relative">
          <div className="w-24 h-24 border-4 border-white/10 rounded-full"></div>
          <div className="absolute top-0 left-0 w-24 h-24 border-4 border-primary-500 rounded-full border-t-transparent animate-spin"></div>
        </div>
        <p className="text-xl font-medium text-gray-300 animate-pulse">Running full extraction pipeline...</p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col items-center justify-center min-h-[calc(100vh-88px)] w-full max-w-3xl mx-auto">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full glass-panel p-10 rounded-3xl"
      >
        <div className="text-center mb-10">
          <h2 className="text-4xl font-black mb-4 bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">Analyze & Compare</h2>
          <p className="text-gray-400 text-lg">Enter a YouTube Video and an Instagram Reel to extract metadata, transcribe audio, and query performance insights.</p>
        </div>

        <form onSubmit={handleAnalyze} className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-semibold text-gray-300 flex items-center gap-2">
              <Youtube size={16} className="text-red-500" /> YouTube Video URL
            </label>
            <input 
              type="url" 
              required
              placeholder="https://youtube.com/watch?v=..."
              className="w-full glass-input px-4 py-3"
              value={ytUrl}
              onChange={e => setYtUrl(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-semibold text-gray-300 flex items-center gap-2">
              <Instagram size={16} className="text-pink-500" /> Instagram Reel URL
            </label>
            <input 
              type="url" 
              required
              placeholder="https://instagram.com/reel/..."
              className="w-full glass-input px-4 py-3"
              value={igUrl}
              onChange={e => setIgUrl(e.target.value)}
            />
          </div>

          {error && <p className="text-red-400 text-sm font-medium p-3 bg-red-400/10 rounded-lg">{error}</p>}

          <button type="submit" className="w-full btn-primary py-4 text-lg mt-4 flex items-center justify-center gap-2">
            Analyze Content <ArrowRight size={20} />
          </button>
        </form>
      </motion.div>
    </div>
  );
};

export default HomePage;
