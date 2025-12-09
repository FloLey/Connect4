import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useGameSocket } from '../hooks/useGameSocket';
import GameBoard from '../components/GameBoard';
import { getGame, getPendingHumanGames } from '../api/client';
import { Brain, Cpu, User, Play, Pause, SkipBack, SkipForward, Eye, AlertCircle, ChevronLeft, ChevronRight, Gamepad2 } from 'lucide-react';

const Arena = () => {
  const { id } = useParams();
  const gameId = parseInt(id);
  const navigate = useNavigate(); // Hook for navigation
  const [meta, setMeta] = useState(null);
  const [replayStep, setReplayStep] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const { gameState, isThinking, isConnected, sendMove } = useGameSocket(gameId);
  
  // NEW: State for actionable games
  const [actionableGames, setActionableGames] = useState([]);

  useEffect(() => {
    getGame(gameId).then((game) => {
      setMeta(game);
      // If game is completed, enter replay mode
      if (game.status === 'COMPLETED' || game.status === 'DRAW') {
        setReplayStep(game.history ? game.history.length - 1 : 0);
      }
    }).catch(console.error);
  }, [gameId]);

  // NEW: Watch for Live Game Completion to trigger Replay Mode
  useEffect(() => {
    if (gameState && (gameState.status === 'COMPLETED' || gameState.status === 'DRAW')) {
      // Game just finished live. Fetch full history to enable replay.
      getGame(gameId).then((fullGame) => {
        setMeta(fullGame); // This updates status and populates history
        setReplayStep(fullGame.history.length - 1); // Jump to end
      }).catch(console.error);
    }
  }, [gameState?.status, gameId]);

  // Auto-play replay
  useEffect(() => {
    if (isPlaying && meta?.history && replayStep < meta.history.length - 1) {
      const timer = setTimeout(() => {
        setReplayStep(prev => prev + 1);
      }, 1500);
      return () => clearTimeout(timer);
    } else {
      setIsPlaying(false);
    }
  }, [isPlaying, replayStep, meta?.history]);

  // NEW: Fetch actionable games list
  const fetchActionableGames = async () => {
    try {
      const ids = await getPendingHumanGames();
      console.log("Queue from API:", ids); // <--- Add this debug log
      setActionableGames(ids);
    } catch (error) {
      console.error("Failed to fetch pending games", error);
    }
  };

  // --- FIX START: Added status and winner to dependencies ---
  // Initial fetch + fetch whenever game state updates (e.g. after I move, win, or lose)
  useEffect(() => {
    fetchActionableGames();
  }, [gameState?.currentTurn, gameState?.status, gameState?.winner, gameId]);
  // --- FIX END ---

  // Helper to calculate Next/Prev
  const handleNavigate = (direction) => {
    if (actionableGames.length === 0) return;

    const currentIndex = actionableGames.indexOf(gameId);
    let nextId;

    if (currentIndex === -1) {
      // If current game is not in the list (e.g. I just played), go to the first one
      nextId = actionableGames[0];
    } else {
      // Circular navigation
      if (direction === 'next') {
        const nextIndex = (currentIndex + 1) % actionableGames.length;
        nextId = actionableGames[nextIndex];
      } else {
        const prevIndex = (currentIndex - 1 + actionableGames.length) % actionableGames.length;
        nextId = actionableGames[prevIndex];
      }
    }
    
    // Only navigate if it's a different game
    if (nextId !== gameId) {
      navigate(`/game/${nextId}`);
    }
  };

  // Helper function to reconstruct board at specific step
  const reconstructBoard = (history, step) => {
    // Initialize empty 6x7 board
    const board = Array(6).fill().map(() => Array(7).fill(0));
    
    // Replay moves up to step
    for (let i = 0; i <= step && i < history.length; i++) {
      const move = history[i];
      const col = move.column;
      
      // Find the lowest empty row in this column
      for (let row = 5; row >= 0; row--) {
        if (board[row][col] === 0) {
          board[row][col] = move.player;
          break;
        }
      }
    }
    
    return board;
  };

  const isSpectatorMode = meta?.player_1_type !== 'human' && meta?.player_2_type !== 'human';
  const isReplayMode = meta?.status === 'COMPLETED' || meta?.status === 'DRAW';
  const isLiveMode = !isReplayMode && isConnected;

  const handleColumnClick = (colIndex) => {
    // Disable interactions in replay mode or spectator mode
    if (isReplayMode || isSpectatorMode) return;
    
    if (gameState && !gameState.winner && !isThinking) {
      sendMove(colIndex);
    }
  };

  if (!gameState && !meta) return <div className="text-center p-10 text-gray-500">Connecting to Arena...</div>;

  const p1Name = meta?.player_1_type || "Player 1";
  const p2Name = meta?.player_2_type || "Player 2";
  
  // Use replay state if in replay mode
  let displayBoard, currentTurn, status, winner, lastMove;
  
  if (isReplayMode && meta?.history) {
    displayBoard = reconstructBoard(meta.history, replayStep);
    currentTurn = replayStep < meta.history.length ? meta.history[replayStep]?.player : 1;
    status = meta.status;
    winner = meta.winner;
    lastMove = replayStep >= 0 ? meta.history[replayStep] : null;
  } else {
    displayBoard = gameState?.board;
    currentTurn = gameState?.currentTurn || 1;
    status = gameState?.status || "LOADING";
    winner = gameState?.winner;
    lastMove = gameState?.lastMove;
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Sidebar */}
      <div className="lg:col-span-1 space-y-6">
        
        {/* --- NEW: Quick Navigator Card --- */}
        {actionableGames.length > 0 && (
          <div className="bg-brand-600 dark:bg-brand-900 rounded-xl p-4 text-white shadow-lg border border-brand-500">
            <div className="flex items-center gap-2 mb-3 opacity-90">
               <Gamepad2 size={18} />
               <span className="text-sm font-bold uppercase tracking-wider">Your Turn Queue</span>
            </div>
            
            <div className="flex items-center justify-between gap-2">
              <button 
                onClick={() => handleNavigate('prev')}
                className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition"
                title="Previous Active Game"
              >
                <ChevronLeft size={20} />
              </button>
              
              <div className="text-center">
                 <div className="text-2xl font-bold">{actionableGames.length}</div>
                 <div className="text-xs opacity-75">Games Waiting</div>
              </div>

              <button 
                onClick={() => handleNavigate('next')}
                className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition"
                title="Next Active Game"
              >
                <ChevronRight size={20} />
              </button>
            </div>
            
            {!actionableGames.includes(gameId) && (
               <div className="mt-3 text-center">
                 <button 
                   onClick={() => navigate(`/game/${actionableGames[0]}`)}
                   className="text-xs bg-white text-brand-600 px-3 py-1 rounded-full font-bold shadow-sm hover:bg-gray-100"
                 >
                   Jump to Game #{actionableGames[0]}
                 </button>
               </div>
            )}
          </div>
        )}
        {/* ----------------------------------- */}
        
        {/* Match Info Card */}
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 shadow-sm">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-lg font-bold text-gray-900 dark:text-white">Match #{gameId}</h2>
            <div className="flex gap-2">
              {isSpectatorMode && isLiveMode && (
                <span className="px-2 py-1 rounded-md text-xs font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 flex items-center gap-1">
                  <Eye size={12} /> Spectating
                </span>
              )}
              {isReplayMode && (
                <span className="px-2 py-1 rounded-md text-xs font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300">
                  Replay
                </span>
              )}
              {isLiveMode && (
                <span className={`px-2 py-1 rounded-md text-xs font-medium ${isConnected ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'}`}>
                  {isConnected ? 'LIVE' : 'OFFLINE'}
                </span>
              )}
            </div>
          </div>

          {/* Players */}
          <div className="space-y-3">
            <div className={`p-4 rounded-lg border transition-all ${currentTurn === 1 ? 'bg-red-50 border-red-200 dark:bg-red-900/10 dark:border-red-900/50 shadow-sm' : 'bg-gray-50 border-gray-100 dark:bg-gray-800 dark:border-gray-700 opacity-60'}`}>
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-red-500 shadow-sm"></div>
                <div className="font-medium text-gray-900 dark:text-white flex items-center gap-2">
                  {p1Name === 'human' ? <User size={16}/> : <Cpu size={16}/>} {p1Name}
                </div>
              </div>
            </div>
            <div className={`p-4 rounded-lg border transition-all ${currentTurn === 2 ? 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/10 dark:border-yellow-900/50 shadow-sm' : 'bg-gray-50 border-gray-100 dark:bg-gray-800 dark:border-gray-700 opacity-60'}`}>
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-yellow-400 shadow-sm"></div>
                <div className="font-medium text-gray-900 dark:text-white flex items-center gap-2">
                  {p2Name === 'human' ? <User size={16}/> : <Cpu size={16}/>} {p2Name}
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-6 border-t border-gray-100 dark:border-gray-800 text-center">
             {winner ? (
                <div className="p-3 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 font-bold rounded-lg border border-green-100 dark:border-green-900/30">
                  🏆 {winner === 1 ? p1Name : p2Name} Wins!
                </div>
             ) : (
                <div className="text-gray-500 dark:text-gray-400 text-sm">
                  Status: <span className="text-gray-900 dark:text-white font-medium">{status}</span>
                </div>
             )}
          </div>
        </div>

        {/* AI Reasoning Card */}
        {lastMove?.reasoning && (
          <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 shadow-sm flex flex-col h-[300px]">
            <div className="flex items-center gap-2 mb-4 text-brand-600 dark:text-brand-400">
              <Brain size={20} /><h3 className="font-bold">AI Thought Process</h3>
            </div>
            
            <div className={`flex-1 p-4 rounded-lg text-sm leading-relaxed border overflow-y-auto ${
              lastMove.reasoning.startsWith('⚠️') 
                ? 'bg-red-50 border-red-100 text-red-700 dark:bg-red-900/20 dark:border-red-900/50 dark:text-red-300' 
                : 'bg-gray-50 border-gray-100 text-gray-700 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300'
            }`}>
              <p className="whitespace-pre-wrap">{lastMove.reasoning}</p>
            </div>
            
            <div className="mt-3 text-right text-xs text-gray-400 font-mono">
              Tokens: {lastMove.input_tokens || 0} in / {lastMove.output_tokens || 0} out
            </div>
          </div>
        )}

        {/* Replay Controls */}
        {isReplayMode && meta?.history && (
          <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 shadow-sm">
            <div className="flex items-center gap-2 mb-4 text-purple-600 dark:text-purple-400">
              <Play size={20} />
              <h3 className="font-bold">Replay Controls</h3>
            </div>
            
            <div className="space-y-6">
              <div className="flex items-center justify-center gap-4">
                <button onClick={() => setReplayStep(0)} disabled={replayStep === 0} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full transition disabled:opacity-30">
                  <SkipBack size={20} className="text-gray-600 dark:text-gray-300" />
                </button>
                
                <button onClick={() => setIsPlaying(!isPlaying)} disabled={replayStep >= meta.history.length - 1} className="p-4 bg-purple-600 hover:bg-purple-700 text-white rounded-full shadow-md transition disabled:opacity-50 disabled:cursor-not-allowed">
                  {isPlaying ? <Pause size={24} /> : <Play size={24} className="ml-1" />}
                </button>
                
                <button onClick={() => setReplayStep(meta.history.length - 1)} disabled={replayStep >= meta.history.length - 1} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full transition disabled:opacity-30">
                  <SkipForward size={20} className="text-gray-600 dark:text-gray-300" />
                </button>
              </div>
              
              <div className="space-y-2">
                <input
                  type="range"
                  min="0"
                  max={meta.history.length - 1}
                  value={replayStep}
                  onChange={(e) => setReplayStep(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
                />
                <div className="flex justify-between text-xs text-gray-500 font-medium">
                  <span>Start</span>
                  <span>Move {replayStep + 1} / {meta.history.length}</span>
                  <span>End</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Game Board Area */}
      <div className="lg:col-span-2">
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-8 flex flex-col items-center relative min-h-[600px] justify-center shadow-sm">
           
           {/* Overlays */}
           {isThinking && isLiveMode && (
             <div className="absolute top-6 right-6 z-20 bg-white/90 dark:bg-gray-800/90 backdrop-blur text-yellow-600 dark:text-yellow-400 px-4 py-2 rounded-full flex items-center gap-2 border border-yellow-200 dark:border-yellow-900/50 shadow-lg">
               <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
               <span className="text-sm font-bold">AI Thinking...</span>
             </div>
           )}
           
           {displayBoard ? (
             <GameBoard 
                board={displayBoard} 
                onColumnClick={handleColumnClick} 
                currentTurn={currentTurn} 
                isHumanTurn={!isSpectatorMode && !isReplayMode} 
                winner={winner} 
             />
           ) : (
             <div className="text-gray-400 flex flex-col items-center gap-3">
                <div className="w-8 h-8 border-4 border-gray-300 border-t-brand-600 rounded-full animate-spin"></div>
                <p>Loading Game State...</p>
             </div>
           )}
        </div>
      </div>
    </div>
  );
};

export default Arena;