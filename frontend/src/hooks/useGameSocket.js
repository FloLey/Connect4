import { useEffect, useState, useRef } from 'react';
import { useDatabase } from '../context/DatabaseContext';

const PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = import.meta.env.VITE_API_URL 
  ? import.meta.env.VITE_API_URL.replace('http', 'ws') 
  : `${PROTOCOL}//${window.location.hostname}:8000`;

export const useGameSocket = (gameId) => {
  const { dbEnv } = useDatabase();
  const [gameState, setGameState] = useState(null);
  const [isThinking, setIsThinking] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    // Clear previous game state immediately when gameId changes
    // This prevents "flickering" of the old game while loading the new one
    setGameState(null);
    setIsThinking(false);
    
    if (!gameId) return;

    // Retrieve token from local storage
    const token = localStorage.getItem(`game_${gameId}_token`);
    
    // Add to WS URL
    const ws = new WebSocket(`${WS_URL}/games/${gameId}/ws?env=${dbEnv}${token ? `&token=${token}` : ''}`);
    socketRef.current = ws;

    ws.onopen = () => {
      console.log('Connected to Game WS');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'UPDATE') {
        setGameState({
          board: data.board,
          // --- FIX START: Match backend camelCase keys ---
          currentTurn: data.currentTurn, 
          winner: data.winner,
          status: data.status,
          lastMove: data.lastMove 
          // --- FIX END ---
        });
      } else if (data.type === 'THINKING_START') {
        setIsThinking(true);
      } else if (data.type === 'THINKING_END') {
        setIsThinking(false);
      }
    };

    ws.onclose = () => setIsConnected(false);

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [gameId, dbEnv]);

  const sendMove = (colIndex) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({
        action: 'MOVE',
        column: colIndex
      }));
    }
  };

  return { gameState, isThinking, isConnected, sendMove };
};