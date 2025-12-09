import React from 'react';
import { clsx } from 'clsx';

const MiniGameBoard = ({ board }) => {
  if (!board) return null;

  return (
    <div className="bg-blue-600 dark:bg-blue-700 p-2 rounded-lg shadow-inner border-2 border-blue-700 dark:border-blue-800">
      <div className="grid grid-rows-6 gap-1">
        {board.map((row, rowIndex) => (
          <div key={rowIndex} className="grid grid-cols-7 gap-1">
            {row.map((cell, colIndex) => (
              <div 
                key={`${rowIndex}-${colIndex}`}
                className={clsx(
                  "w-3 h-3 md:w-4 md:h-4 rounded-full flex items-center justify-center shadow-inner",
                  cell === 0 
                    ? "bg-blue-800 dark:bg-gray-900" 
                    : cell === 1 
                      ? "bg-red-500 shadow-sm" 
                      : "bg-yellow-400 shadow-sm"
                )}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export default MiniGameBoard;