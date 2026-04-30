import { useEffect, useRef, useState } from 'react';
import { useDatabase } from '../context/DatabaseContext';

const PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL.replace('http', 'ws')
  : `${PROTOCOL}//${window.location.hostname}:8000`;

// Exponential backoff schedule (ms): 1s, 2s, 4s, 8s, 16s, then 30s cap.
const BACKOFF_BASE_MS = 1000;
const BACKOFF_MAX_MS = 30000;
const BACKOFF_JITTER = 0.2; // ±20%

const computeBackoff = (attempt) => {
  const exp = Math.min(BACKOFF_MAX_MS, BACKOFF_BASE_MS * 2 ** attempt);
  const jitter = exp * BACKOFF_JITTER * (Math.random() * 2 - 1);
  return Math.max(BACKOFF_BASE_MS, Math.round(exp + jitter));
};

export const useGameSocket = (gameId) => {
  const { dbEnv } = useDatabase();
  const [gameState, setGameState] = useState(null);
  const [isThinking, setIsThinking] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const [lastError, setLastError] = useState(null);

  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const attemptRef = useRef(0);
  const closedByUserRef = useRef(false);

  useEffect(() => {
    // Reset surface state whenever gameId or dbEnv changes.
    setGameState(null);
    setIsThinking(false);
    setReconnectAttempt(0);
    setLastError(null);
    attemptRef.current = 0;
    closedByUserRef.current = false;

    if (!gameId) return undefined;

    const token = localStorage.getItem(`game_${gameId}_token`);
    const url = `${WS_URL}/games/${gameId}/ws?env=${dbEnv}${token ? `&token=${token}` : ''}`;

    const connect = () => {
      const ws = new WebSocket(url);
      socketRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        attemptRef.current = 0;
        setReconnectAttempt(0);
        setLastError(null);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'UPDATE') {
          setGameState({
            board: data.board,
            currentTurn: data.currentTurn,
            winner: data.winner,
            status: data.status,
            lastMove: data.lastMove,
          });
        } else if (data.type === 'THINKING_START') {
          setIsThinking(true);
        } else if (data.type === 'THINKING_END') {
          setIsThinking(false);
        }
      };

      ws.onerror = (event) => {
        setLastError(event);
      };

      ws.onclose = () => {
        setIsConnected(false);
        if (closedByUserRef.current) return;

        // Schedule a reconnect with backoff.
        const delay = computeBackoff(attemptRef.current);
        attemptRef.current += 1;
        setReconnectAttempt(attemptRef.current);
        reconnectTimerRef.current = setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      closedByUserRef.current = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [gameId, dbEnv]);

  const sendMove = (colIndex) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({
        action: 'MOVE',
        column: colIndex,
      }));
    }
  };

  return { gameState, isThinking, isConnected, reconnectAttempt, lastError, sendMove };
};
