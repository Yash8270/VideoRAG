import React from 'react';
import { motion } from 'framer-motion';

const MessageBubble = ({ message }) => {
  const isUser = message.role === 'user';
  
  // Format citations gracefully if AI message contains [Video ID: ..., Chunk: ...]
  const formatTextWithCitations = (text) => {
    if (isUser) return text;
    
    // Split by citation brackets [Video ID: X, Chunk: Y]
    const citationRegex = /\[Video ID: ([^,]+),\s*Chunk:\s*(\d+)\]/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = citationRegex.exec(text)) !== null) {
      // Add text before citation
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      // Add styled citation badge
      parts.push(
        <span key={match.index} className="inline-flex items-center px-2 py-0.5 mx-1 rounded text-xs font-semibold bg-primary-500/20 text-primary-300 border border-primary-500/30 cursor-pointer hover:bg-primary-500/40 transition-colors" title={`Source: ${match[1]}, Chunk ${match[2]}`}>
          📄 {match[2]}
        </span>
      );
      lastIndex = citationRegex.lastIndex;
    }
    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }
    
    return parts.length > 0 ? parts : text;
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
    >
      <div className={`max-w-[85%] rounded-2xl px-5 py-3 shadow-lg ${
        isUser 
          ? 'bg-gradient-to-br from-primary-600 to-accent-600 text-white rounded-br-none' 
          : 'glass-panel text-gray-200 rounded-bl-none border-l-2 border-l-primary-500'
      }`}>
        <p className="text-sm leading-relaxed whitespace-pre-wrap">
          {formatTextWithCitations(message.content)}
        </p>
      </div>
    </motion.div>
  );
};

export default MessageBubble;
