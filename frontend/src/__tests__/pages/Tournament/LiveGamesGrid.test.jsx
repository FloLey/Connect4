import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

// Stub useGameSocket so the per-card sockets in LiveGameCard don't open real
// WebSockets and don't require Database/ToastProvider context just for these
// layout-focused tests.
vi.mock('../../../hooks/useGameSocket', () => ({
  useGameSocket: () => ({
    gameState: null,
    isThinking: false,
    isConnected: false,
    reconnectAttempt: 0,
    lastError: null,
    sendMove: () => {},
  }),
}));

import LiveGamesGrid from '../../../pages/Tournament/LiveGamesGrid';

const board = Array.from({ length: 6 }, () => Array.from({ length: 7 }, () => 0));

const games = [
  {
    id: 5,
    player_1: 'alpha',
    player_2: 'beta',
    move_count: 3,
    status: 'IN_PROGRESS',
    board,
  },
  {
    id: 2,
    player_1: 'gamma',
    player_2: 'delta',
    move_count: 1,
    status: 'IN_PROGRESS',
    board,
  },
];

describe('LiveGamesGrid', () => {
  it('renders the empty-state message when no games are running', () => {
    render(
      <MemoryRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <LiveGamesGrid tournamentStatus="SETUP" activeGames={[]} />
      </MemoryRouter>
    );
    expect(screen.getByText(/Waiting to start/i)).toBeInTheDocument();
  });

  it('renders one card per game in the order received from props', () => {
    render(
      <MemoryRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <LiveGamesGrid tournamentStatus="IN_PROGRESS" activeGames={games} />
      </MemoryRouter>
    );

    const ids = screen.getAllByText(/^#\d+$/).map((el) => el.textContent);
    expect(ids).toEqual(['#5', '#2']);
  });

  it('flags snoozed games with the rate-limit pill', () => {
    const snoozed = [
      {
        ...games[0],
        status: 'PAUSED',
        retry_after: new Date(Date.now() + 60_000).toISOString(),
      },
    ];
    render(
      <MemoryRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <LiveGamesGrid tournamentStatus="IN_PROGRESS" activeGames={snoozed} />
      </MemoryRouter>
    );
    expect(screen.getByText(/Rate Limit Cooldown/i)).toBeInTheDocument();
    expect(screen.getByText(/Snoozed/i)).toBeInTheDocument();
  });
});
