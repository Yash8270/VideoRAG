import React, { useState, useRef, useEffect, useContext } from 'react';
import { Send, Loader2, Bot } from 'lucide-react';
import MessageBubble from './MessageBubble';
import VideoContext from '../../context/VideoContext';

const ChatPanel = () => {
  const { chatWithVideos, sessionId, analysisData } = useContext(VideoContext);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your VideoRAG AI. Ask me to compare hooks, summarize insights, or suggest improvements based on the analyzed videos.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollContainerRef = useRef(null);

  const scrollToBottom = () => {
    if (scrollContainerRef.current) {
      const { scrollHeight } = scrollContainerRef.current;
      scrollContainerRef.current.scrollTo({
        top: scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userText = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userText }]);
    setIsLoading(true);

    // Extract current video IDs from the active analysis pair
    const videoIds = [];
    if (analysisData?.youtube?.input?.video_id) videoIds.push(analysisData.youtube.input.video_id);
    if (analysisData?.instagram?.input?.video_id) videoIds.push(analysisData.instagram.input.video_id);

    try {
      const response = await chatWithVideos(userText, sessionId, videoIds);
      setMessages(prev => [...prev, { role: 'assistant', content: response.answer }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error connecting to the RAG backend.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full w-full glass-panel rounded-2xl overflow-hidden border-t border-white/10 mt-6 relative">
      {/* Header */}
      <div className="bg-dark-800/80 px-6 py-4 flex items-center gap-3 border-b border-white/5">
        <div className="bg-primary-500/20 p-2 rounded-lg">
          <Bot size={20} className="text-primary-500" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-white tracking-wide">AI Strategist</h2>
          <p className="text-xs text-gray-400">Powered by Gemini 2.5 Flash & HuggingFace</p>
        </div>
      </div>

      {/* Messages Area */}
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto p-6 space-y-2 max-h-[60vh]">
        {messages.map((msg, idx) => (
          <MessageBubble key={idx} message={msg} />
        ))}
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="glass-panel rounded-2xl rounded-bl-none px-5 py-4 border-l-2 border-l-primary-500 flex items-center space-x-2">
              <Loader2 size={16} className="text-primary-500 animate-spin" />
              <span className="text-xs text-gray-400">Analyzing transcripts...</span>
            </div>
          </div>
        )}
        )}
      </div>

      {/* Input Area */}
      <form onSubmit={handleSend} className="p-4 bg-dark-900/50 border-t border-white/5">
        <div className="relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Why did Video A perform better?"
            className="w-full glass-input py-4 pl-6 pr-14 text-sm text-white rounded-full shadow-inner"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="absolute right-2 p-2 bg-primary-600 hover:bg-primary-500 text-white rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
          >
            <Send size={18} className={input.trim() ? "ml-0.5" : ""} />
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatPanel;
