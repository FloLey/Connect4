import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import React from 'react';

import { useGameSocket } from '../../hooks/useGameSocket';
import { DatabaseProvider } from '../../context/DatabaseContext';
import { ToastProvider } from '../../context/ToastContext';

class MockWebSocket {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.readyState = 0; // CONNECTING
    this.sent = [];
    MockWebSocket.instances.push(this);
  }
  send(payload) {
    this.sent.push(payload);
  }
  close() {
    this.readyState = 3;
    this.onclose && this.onclose();
  }
  // helpers to drive the socket from tests
  _open() {
    this.readyState = 1;
    this.onopen && this.onopen();
  }
  _message(data) {
    this.onmessage && this.onmessage({ data: JSON.stringify(data) });
  }
  _close() {
    // Simulate the server-initiated close (does NOT touch closedByUserRef).
    this.readyState = 3;
    this.onclose && this.onclose();
  }
}
MockWebSocket.OPEN = 1;

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.stubGlobal('WebSocket', MockWebSocket);
  // Math.random is used for backoff jitter; pin it to remove flakiness.
  vi.spyOn(Math, 'random').mockReturnValue(0.5);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const wrapper = ({ children }) =>
  React.createElement(
    ToastProvider,
    null,
    React.createElement(DatabaseProvider, null, children)
  );

describe('useGameSocket', () => {
  it('opens a socket and sets isConnected on open', async () => {
    const { result } = renderHook(() => useGameSocket(42), { wrapper });

    expect(MockWebSocket.instances).toHaveLength(1);
    expect(MockWebSocket.instances[0].url).toContain('/games/42/ws');

    act(() => MockWebSocket.instances[0]._open());

    await waitFor(() => expect(result.current.isConnected).toBe(true));
    expect(result.current.reconnectAttempt).toBe(0);
  });

  it('updates gameState when an UPDATE message arrives', async () => {
    const { result } = renderHook(() => useGameSocket(7), { wrapper });
    const ws = MockWebSocket.instances[0];
    act(() => ws._open());

    act(() =>
      ws._message({
        type: 'UPDATE',
        board: [[0]],
        currentTurn: 2,
        winner: null,
        status: 'IN_PROGRESS',
        lastMove: { column: 3 },
      })
    );

    await waitFor(() => expect(result.current.gameState).not.toBeNull());
    expect(result.current.gameState.currentTurn).toBe(2);
    expect(result.current.gameState.lastMove.column).toBe(3);
  });

  it('toggles isThinking on THINKING_START / THINKING_END', async () => {
    const { result } = renderHook(() => useGameSocket(1), { wrapper });
    const ws = MockWebSocket.instances[0];
    act(() => ws._open());

    act(() => ws._message({ type: 'THINKING_START' }));
    await waitFor(() => expect(result.current.isThinking).toBe(true));

    act(() => ws._message({ type: 'THINKING_END' }));
    await waitFor(() => expect(result.current.isThinking).toBe(false));
  });

  it('sendMove serialises the column into a MOVE action', () => {
    const { result } = renderHook(() => useGameSocket(99), { wrapper });
    const ws = MockWebSocket.instances[0];
    act(() => ws._open());

    act(() => result.current.sendMove(4));
    expect(ws.sent).toEqual([JSON.stringify({ action: 'MOVE', column: 4 })]);
  });

  it('does not send when the socket is not open', () => {
    const { result } = renderHook(() => useGameSocket(99), { wrapper });
    // Socket is created but never opened; readyState stays 0.
    act(() => result.current.sendMove(4));
    expect(MockWebSocket.instances[0].sent).toEqual([]);
  });

  it('clears game state when gameId changes', async () => {
    const { result, rerender } = renderHook(({ id }) => useGameSocket(id), {
      wrapper,
      initialProps: { id: 1 },
    });
    act(() => MockWebSocket.instances[0]._open());
    act(() =>
      MockWebSocket.instances[0]._message({
        type: 'UPDATE',
        board: [[1]],
        currentTurn: 2,
        winner: null,
        status: 'IN_PROGRESS',
        lastMove: null,
      })
    );
    await waitFor(() => expect(result.current.gameState).not.toBeNull());

    rerender({ id: 2 });
    expect(result.current.gameState).toBeNull();
    expect(MockWebSocket.instances).toHaveLength(2);
    expect(MockWebSocket.instances[1].url).toContain('/games/2/ws');
  });

  // -- Reconnect / backoff -------------------------------------------------

  describe('reconnect with backoff', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });
    afterEach(() => {
      vi.useRealTimers();
    });

    it('reconnects after a server-initiated close, incrementing reconnectAttempt', async () => {
      const { result } = renderHook(() => useGameSocket(11), { wrapper });
      const first = MockWebSocket.instances[0];
      act(() => first._open());
      expect(result.current.reconnectAttempt).toBe(0);

      // Server drops the connection.
      act(() => first._close());
      expect(result.current.isConnected).toBe(false);
      expect(result.current.reconnectAttempt).toBe(1);

      // Backoff = 1000ms (attempt=0 → base, jitter pinned to 0).
      act(() => {
        vi.advanceTimersByTime(1500);
      });
      expect(MockWebSocket.instances).toHaveLength(2);
      expect(MockWebSocket.instances[1].url).toBe(first.url);
    });

    it('resets reconnectAttempt when the new socket opens successfully', async () => {
      const { result } = renderHook(() => useGameSocket(12), { wrapper });
      const first = MockWebSocket.instances[0];
      act(() => first._open());

      // Drop, reconnect, drop again so attempt climbs to 2.
      act(() => first._close());
      act(() => vi.advanceTimersByTime(1500));
      const second = MockWebSocket.instances[1];
      act(() => second._close());
      act(() => vi.advanceTimersByTime(3000));
      expect(MockWebSocket.instances).toHaveLength(3);
      expect(result.current.reconnectAttempt).toBe(2);

      // The third socket finally opens.
      const third = MockWebSocket.instances[2];
      act(() => third._open());
      expect(result.current.reconnectAttempt).toBe(0);
      expect(result.current.isConnected).toBe(true);
    });

    it('does not reconnect when the hook unmounts', () => {
      const { unmount } = renderHook(() => useGameSocket(13), { wrapper });
      const first = MockWebSocket.instances[0];
      act(() => first._open());

      unmount();
      // Cleanup closes the socket; that triggers onclose, but closedByUserRef
      // should suppress the reconnect.
      act(() => vi.advanceTimersByTime(60000));
      expect(MockWebSocket.instances).toHaveLength(1);
    });
  });
});
