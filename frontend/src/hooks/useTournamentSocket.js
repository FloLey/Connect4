import { useEffect, useRef } from 'react';

import { useDatabase } from '../context/DatabaseContext';

const PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL.replace('http', 'ws')
  : `${PROTOCOL}//${window.location.hostname}:8000`;

/**
 * Subscribe to a tournament's control-plane WebSocket. ``onEvent`` receives
 * ``{type: 'GAME_STARTED', game}`` or ``{type: 'GAME_COMPLETED', game_id, winner}``.
 *
 * Auto-reconnects with the same backoff schedule as ``useGameSocket`` since
 * the dropped-connection failure modes are identical.
 */
export const useTournamentSocket = (tournamentId, onEvent) => {
  const { dbEnv } = useDatabase();
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const closedByUserRef = useRef(false);
  const attemptRef = useRef(0);

  useEffect(() => {
    if (!tournamentId) return undefined;

    closedByUserRef.current = false;
    attemptRef.current = 0;
    const url = `${WS_URL}/tournament/${tournamentId}/ws?env=${dbEnv}`;

    const connect = () => {
      const ws = new WebSocket(url);
      socketRef.current = ws;

      ws.onopen = () => {
        attemptRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onEventRef.current?.(data);
        } catch {
          // Drop malformed payloads silently.
        }
      };

      ws.onclose = () => {
        if (closedByUserRef.current) return;
        const delay = Math.min(30000, 1000 * 2 ** attemptRef.current);
        attemptRef.current += 1;
        reconnectTimerRef.current = setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      closedByUserRef.current = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (socketRef.current) socketRef.current.close();
    };
  }, [tournamentId, dbEnv]);
};
