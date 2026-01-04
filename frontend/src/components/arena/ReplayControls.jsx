import React from 'react';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';

const ReplayControls = ({ step, total, isPlaying, onSeek, onPlayPause }) => {
  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 shadow-sm">
      <div className="flex items-center gap-2 mb-4 text-purple-600 dark:text-purple-400">
        <Play size={20} />
        <h3 className="font-bold">Replay Controls</h3>
      </div>
      
      <div className="space-y-6">
        <div className="flex items-center justify-center gap-4">
          <button 
            onClick={() => onSeek(0)} 
            disabled={step === 0} 
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full transition disabled:opacity-30"
          >
            <SkipBack size={20} className="text-gray-600 dark:text-gray-300" />
          </button>
          
          <button 
            onClick={onPlayPause} 
            disabled={step >= total - 1} 
            className="p-4 bg-purple-600 hover:bg-purple-700 text-white rounded-full shadow-md transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isPlaying ? <Pause size={24} /> : <Play size={24} className="ml-1" />}
          </button>
          
          <button 
            onClick={() => onSeek(total - 1)} 
            disabled={step >= total - 1} 
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full transition disabled:opacity-30"
          >
            <SkipForward size={20} className="text-gray-600 dark:text-gray-300" />
          </button>
        </div>
        
        <div className="space-y-2">
          <input
            type="range"
            min="0"
            max={total - 1}
            value={step}
            onChange={(e) => onSeek(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
          />
          <div className="flex justify-between text-xs text-gray-500 font-medium">
            <span>Start</span>
            <span>Move {step + 1} / {total}</span>
            <span>End</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReplayControls;