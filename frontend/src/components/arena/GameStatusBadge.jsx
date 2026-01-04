import React from 'react';
import { Eye } from 'lucide-react';

const GameStatusBadge = ({ type, isConnected = true }) => {
  const badgeConfig = {
    spectating: {
      text: 'Spectating',
      icon: <Eye size={12} />,
      bg: 'bg-blue-100 dark:bg-blue-900/30',
      textColor: 'text-blue-700 dark:text-blue-300'
    },
    replay: {
      text: 'Replay',
      icon: null,
      bg: 'bg-purple-100 dark:bg-purple-900/30',
      textColor: 'text-purple-700 dark:text-purple-300'
    },
    live: {
      text: isConnected ? 'LIVE' : 'OFFLINE',
      icon: null,
      bg: isConnected ? 'bg-green-100 dark:bg-green-900/30' : 'bg-red-100 dark:bg-red-900/30',
      textColor: isConnected ? 'text-green-700 dark:text-green-300' : 'text-red-700 dark:text-red-300'
    }
  };

  const config = badgeConfig[type];
  if (!config) return null;

  return (
    <span className={`px-2 py-1 rounded-md text-xs font-medium ${config.bg} ${config.textColor} flex items-center gap-1`}>
      {config.icon}
      {config.text}
    </span>
  );
};

export default GameStatusBadge;