import React from 'react';
import { motion } from 'framer-motion';
import { Heart, MessageCircle, Eye, Calendar, Clock, Users } from 'lucide-react';

const fmt = (n) => {
  if (n == null) return null;
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000)     return (n / 1_000).toFixed(1)     + 'K';
  return n.toLocaleString();
};

const StatBadge = ({ icon: Icon, value, label }) => (
  <div className="flex flex-col items-center justify-center p-3 rounded-xl bg-white/5 border border-white/5 backdrop-blur-sm">
    <div className="flex items-center space-x-2 text-gray-300 mb-1">
      <Icon size={16} className="text-primary-500" />
      <span className="text-xs uppercase tracking-wider font-semibold">{label}</span>
    </div>
    <span className="text-lg font-bold text-white">{value ?? 'N/A'}</span>
  </div>
);

const VideoCard = ({ platform, data }) => {
  if (!data) return null;

  const { input, metrics } = data;
  const isYoutube = platform.toLowerCase() === 'youtube';

  // Show N/A instead of 0 when the platform didn't report the metric
  const viewsDisplay     = metrics.has_missing_views    ? 'N/A' : metrics.views_formatted;
  const likesDisplay     = metrics.has_missing_likes    ? 'N/A' : metrics.likes_formatted;
  const commentsDisplay  = metrics.has_missing_comments ? 'N/A' : metrics.comments_formatted;
  const followersDisplay = fmt(input.follower_count) ?? 'N/A';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="glass-panel rounded-2xl overflow-hidden flex flex-col h-full relative group"
    >
      {/* Platform Badge */}
      <div className={`absolute top-4 right-4 px-3 py-1 text-xs font-bold uppercase rounded-full shadow-lg backdrop-blur-md border ${
        isYoutube
          ? 'bg-red-500/20 text-red-400 border-red-500/30'
          : 'bg-pink-500/20 text-pink-400 border-pink-500/30'
      }`}>
        {platform}
      </div>

      {/* Thumbnail Header */}
      <div className="h-48 shrink-0 relative overflow-hidden bg-dark-900">
        <div className="absolute inset-0 bg-gradient-to-t from-dark-800 to-transparent z-10" />
        {isYoutube ? (
          <img
            src={`https://img.youtube.com/vi/${input.video_id}/maxresdefault.jpg`}
            alt={input.title}
            className="w-full h-full object-cover opacity-60 group-hover:opacity-80 transition-opacity duration-500"
            onError={(e) => { e.target.style.display = 'none'; }}
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-tr from-yellow-500 via-pink-500 to-purple-600 opacity-40 group-hover:opacity-60 transition-opacity duration-500" />
        )}
        <div className="absolute bottom-4 left-4 right-4 z-20">
          <h3 className="text-xl font-bold line-clamp-2 leading-tight drop-shadow-lg">{input.title}</h3>
          <p className="text-gray-400 text-sm mt-1">@{input.creator}</p>
        </div>
      </div>

      {/* Stats Body */}
      <div className="p-6 flex-1 flex flex-col">

        {/* Engagement Rate Hero */}
        <div className="mb-6 text-center pb-6 border-b border-white/10">
          <p className="text-sm text-gray-400 mb-1">Engagement Rate</p>
          <div className="text-4xl font-black bg-clip-text text-transparent bg-gradient-to-r from-primary-500 to-accent-500">
            {metrics.engagement_rate_label}
          </div>
          <p className="text-xs text-primary-500 mt-2 font-medium tracking-wide uppercase">
            {metrics.engagement_level} tier
          </p>
        </div>

        {/* Metrics Grid — 2×2 */}
        <div className="grid grid-cols-2 gap-3 mb-6">
          <StatBadge icon={Eye}           value={viewsDisplay}     label="Views"     />
          <StatBadge icon={Heart}         value={likesDisplay}     label="Likes"     />
          <StatBadge icon={MessageCircle} value={commentsDisplay}  label="Comments"  />
          <StatBadge icon={Users}         value={followersDisplay} label={isYoutube ? "Subscribers" : "Followers"} />
        </div>

        {/* Footer */}
        <div className="mt-auto flex justify-between text-xs text-gray-500 px-2">
          <div className="flex items-center gap-1">
            <Calendar size={14} />
            {input.upload_date ? new Date(input.upload_date).toLocaleDateString() : 'Unknown date'}
          </div>
          <div className="flex items-center gap-1">
            <Clock size={14} />
            {input.duration_seconds
              ? `${Math.floor(input.duration_seconds / 60)}:${(input.duration_seconds % 60).toString().padStart(2, '0')}`
              : '--:--'}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default VideoCard;
