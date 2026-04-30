import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

// Each test stubs useGameSocket via the same module path; the stub return value
// changes per test using vi.mocked.
vi.mock('../../../hooks/useGameSocket', () => ({
  useGameSocket: vi.fn(),
}));

import { useGameSocket } from '../../../hooks/useGameSocket';
import LiveGameCard from '../../../pages/Tournament/LiveGameCard';

const baseGame = {
  id: 99,
  player_1: 'alpha',
  player_2: 'beta',
  status: 'IN_PROGRESS',
  move_count: 2,
  board: Array.from({ length: 6 }, () => Array.from({ length: 7 }, () => 0)),
};

const renderCard = (game = baseGame) =>
  render(
    <MemoryRouter
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      <LiveGameCard game={game} tournamentStatus="IN_PROGRESS" />
    </MemoryRouter>
  );

const stubSocket = (overrides = {}) => {
  useGameSocket.mockReturnValue({
    gameState: null,
    isThinking: false,
    isConnected: false,
    reconnectAttempt: 0,
    lastError: null,
    sendMove: () => {},
    ...overrides,
  });
};

describe('LiveGameCard', () => {
  it('falls back to the polled snapshot when no WS state has arrived', () => {
    stubSocket();
    renderCard();
    // move_count from the polled snapshot.
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('alpha')).toBeInTheDocument();
    expect(screen.getByText('beta')).toBeInTheDocument();
    expect(screen.getByText(/Live/i)).toBeInTheDocument();
  });

  it('uses the WS-derived board to compute live move count', () => {
    const board = baseGame.board.map((row) => [...row]);
    board[5][0] = 1;
    board[5][1] = 2;
    board[4][0] = 1;
    stubSocket({
      gameState: {
        board,
        currentTurn: 2,
        winner: null,
        status: 'IN_PROGRESS',
        lastMove: { column: 0 },
      },
    });
    renderCard({ ...baseGame, move_count: 0 }); // snapshot says 0 but socket has 3 placed pieces
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('subscribes to its own game id', () => {
    stubSocket();
    renderCard();
    expect(useGameSocket).toHaveBeenCalledWith(99);
  });

  it('renders the snoozed pill when status is PAUSED with future retry_after', () => {
    stubSocket();
    renderCard({
      ...baseGame,
      status: 'PAUSED',
      retry_after: new Date(Date.now() + 60_000).toISOString(),
    });
    expect(screen.getByText(/Rate Limit Cooldown/i)).toBeInTheDocument();
    expect(screen.getByText(/Snoozed/i)).toBeInTheDocument();
  });
});
