import React from 'react';
import { clsx } from 'clsx';
import { motion } from 'framer-motion';

const GameBoard = ({ board, onColumnClick, currentTurn, isHumanTurn, winner }) => {
  if (!board) return null;

  return (
    <div className="flex flex-col items-center">
      {/* 
         Classic Blue Board 
         - Shadow added for depth
         - Border to separate from the light/dark background container
      */}
      <div className="bg-blue-600 dark:bg-blue-700 p-4 rounded-xl shadow-2xl inline-block border-4 border-blue-700 dark:border-blue-800 relative z-10">
        <div className="grid grid-rows-6 gap-3">
          {board.map((row, rowIndex) => (
            <div key={rowIndex} className="grid grid-cols-7 gap-3">
              {row.map((cell, colIndex) => (
                <div 
                  key={`${rowIndex}-${colIndex}`}
                  onClick={() => isHumanTurn && !winner && onColumnClick(colIndex)}
                  className={clsx(
                    "w-12 h-12 md:w-16 md:h-16 rounded-full flex items-center justify-center relative",
                    // The "Empty Hole" look: Dark blue background with inner shadow to look like a hole
                    "bg-blue-800 dark:bg-gray-900 shadow-[inset_0_4px_6px_rgba(0,0,0,0.4)]",
                    isHumanTurn && !winner ? "cursor-pointer hover:brightness-110" : "cursor-default"
                  )}
                >
                  {/* The Piece */}
                  {cell !== 0 && (
                    <motion.div
                      initial={{ y: -400, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      transition={{ type: "spring", stiffness: 250, damping: 25 }}
                      className={clsx(
                        "w-full h-full rounded-full shadow-md border-4",
                        // Red Piece
                        cell === 1 && "bg-red-500 border-red-600",
                        // Yellow Piece
                        cell === 2 && "bg-yellow-400 border-yellow-500"
                      )}
                    />
                  )}
                  
                  {/* Hover Guide (Phantom Piece) - Optional nice-to-have */}
                  {cell === 0 && isHumanTurn && !winner && (
                     <div className={clsx(
                       "absolute w-full h-full rounded-full opacity-0 hover:opacity-20 transition-opacity",
                       currentTurn === 1 ? "bg-red-500" : "bg-yellow-400"
                     )} />
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      
      {/* Column Indicators */}
      <div className="grid grid-cols-7 gap-3 mt-4 w-full px-4">
        {[0,1,2,3,4,5,6].map(col => (
           <div key={col} className="text-center">
             <span className="text-xs font-mono text-gray-400 dark:text-gray-500 select-none">
               {col + 1}
             </span>
           </div>
        ))}
      </div>
    </div>
  );
};

export default GameBoard;