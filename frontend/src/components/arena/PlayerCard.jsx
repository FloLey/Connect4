import React from 'react';
import { User, Cpu } from 'lucide-react';

const PlayerCard = ({ name, type, isActive, color }) => {
  const colorClasses = {
    red: {
      bg: 'bg-red-50 dark:bg-red-900/10',
      border: 'border-red-200 dark:border-red-900/50',
      dot: 'bg-red-500'
    },
    yellow: {
      bg: 'bg-yellow-50 dark:bg-yellow-900/10',
      border: 'border-yellow-200 dark:border-yellow-900/50',
      dot: 'bg-yellow-400'
    }
  };

  const colors = colorClasses[color] || colorClasses.red;

  return (
    <div className={`p-4 rounded-lg border transition-all ${
      isActive 
        ? `${colors.bg} ${colors.border} shadow-sm` 
        : 'bg-gray-50 border-gray-100 dark:bg-gray-800 dark:border-gray-700 opacity-60'
    }`}>
      <div className="flex items-center gap-3">
        <div className={`w-3 h-3 rounded-full ${colors.dot} shadow-sm`}></div>
        <div className="font-medium text-gray-900 dark:text-white flex items-center gap-2">
          {type === 'human' ? <User size={16}/> : <Cpu size={16}/>} {name}
        </div>
      </div>
    </div>
  );
};

export default PlayerCard;