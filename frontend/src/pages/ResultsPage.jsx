import React, { useContext, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import VideoContext from '../context/VideoContext';
import VideoCard from '../components/video/VideoCard';
import ChatPanel from '../components/chat/ChatPanel';
import { ArrowLeft } from 'lucide-react';

const ResultsPage = () => {
  const { analysisData } = useContext(VideoContext);
  const navigate = useNavigate();

  // Redirect to home if no data is available (e.g., user refreshes page)
  useEffect(() => {
    if (!analysisData) {
      navigate('/');
    }
  }, [analysisData, navigate]);

  if (!analysisData) return null;

  return (
    <div className="flex flex-col w-full gap-6 min-h-min pb-8 h-full">
      
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Analysis Results</h2>
        <button 
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
        >
          <ArrowLeft size={16} /> Analyze Another
        </button>
      </div>

      {/* Top Half: Video Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 shrink-0">
        <VideoCard platform="YouTube" data={analysisData.youtube} />
        <VideoCard platform="Instagram" data={analysisData.instagram} />
      </div>

      {/* Bottom Half: Chat Panel */}
      <div className="flex-1 min-h-[500px] flex flex-col">
        <ChatPanel />
      </div>

    </div>
  );
};

export default ResultsPage;
