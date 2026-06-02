import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Activity } from 'lucide-react';
import HomePage from './pages/HomePage';
import ResultsPage from './pages/ResultsPage';

function App() {
  return (
    <div className="min-h-screen bg-dark-900 text-gray-100 flex flex-col font-sans relative">
      {/* Decorative ambient background */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-primary-600/20 blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-accent-600/20 blur-[120px] rounded-full pointer-events-none" />

      {/* Header */}
      <header className="shrink-0 relative z-10 w-full px-8 py-6 border-b border-white/5 bg-dark-900/40 backdrop-blur-md">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="VideoRAG Logo" className="w-14 h-14 object-contain rounded-xl shadow-[0_0_15px_rgba(139,92,246,0.3)]" />
            <h1 className="text-2xl font-black tracking-tight text-white">Video<span className="text-primary-400">RAG</span></h1>
          </div>
          <p className="text-sm text-gray-400 font-medium">Cross-Platform Engagement Analyzer</p>
        </div>
      </header>

      <main className="flex-1 relative z-10 w-full max-w-7xl mx-auto px-8 py-8 flex flex-col gap-8 h-full">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/results" element={<ResultsPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
