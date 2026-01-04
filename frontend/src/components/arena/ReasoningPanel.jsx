import React from 'react';
import { Brain } from 'lucide-react';

const ReasoningPanel = ({ reasoning, inputTokens = 0, outputTokens = 0 }) => {
  if (!reasoning) return null;

  const isError = reasoning.startsWith('⚠️');
  
  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 shadow-sm flex flex-col h-[300px]">
      <div className="flex items-center gap-2 mb-4 text-brand-600 dark:text-brand-400">
        <Brain size={20} />
        <h3 className="font-bold">AI Thought Process</h3>
      </div>
      
      <div className={`flex-1 p-4 rounded-lg text-sm leading-relaxed border overflow-y-auto ${
        isError
          ? 'bg-red-50 border-red-100 text-red-700 dark:bg-red-900/20 dark:border-red-900/50 dark:text-red-300' 
          : 'bg-gray-50 border-gray-100 text-gray-700 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300'
      }`}>
        <p className="whitespace-pre-wrap">{reasoning}</p>
      </div>
      
      <div className="mt-3 text-right text-xs text-gray-400 font-mono">
        Tokens: {inputTokens} in / {outputTokens} out
      </div>
    </div>
  );
};

export default ReasoningPanel;