import { describe, expect, it, beforeAll } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { http, HttpResponse } from 'msw';

import Statistics from '../../pages/Statistics';
import { AllProviders } from '../../test/wrappers';
import { server } from '../../test/msw-server';

// Recharts uses ResizeObserver internally; jsdom doesn't have one.
beforeAll(() => {
  if (!globalThis.ResizeObserver) {
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
});

const renderStatistics = () =>
  render(
    <AllProviders>
      <MemoryRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Statistics />
      </MemoryRouter>
    </AllProviders>
  );

describe('Statistics page', () => {
  it('renders empty state with no models', async () => {
    renderStatistics();
    await waitFor(() => {
      expect(screen.getByText(/Analytics Center/i)).toBeInTheDocument();
    });
  });

  it('renders model rows from /stats/leaderboard', async () => {
    server.use(
      http.get('http://localhost:8000/stats/leaderboard', () =>
        HttpResponse.json([
          {
            model_name: 'gpt-5',
            rating: 1500.0,
            matches_played: 10,
            wins: 7,
            losses: 2,
            draws: 1,
            mean_time_per_move: 1.2,
            avg_moves_per_game: 8.0,
            mean_tokens_out_per_move: 100,
            total_tokens_out: 1000,
            avg_cost_per_move: 0.0001,
            avg_cost_per_game: 0.0008,
            total_cost: 0.008,
          },
        ])
      )
    );
    renderStatistics();
    await waitFor(() => {
      expect(screen.getAllByText(/gpt-5/).length).toBeGreaterThan(0);
    });
  });
});
