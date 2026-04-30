import { describe, expect, it } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { http, HttpResponse } from 'msw';

import History from '../../pages/History';
import { AllProviders } from '../../test/wrappers';
import { server } from '../../test/msw-server';

const renderHistory = () =>
  render(
    <AllProviders>
      <MemoryRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <History />
      </MemoryRouter>
    </AllProviders>
  );

describe('History page', () => {
  it('shows the empty state when there are no completed games', async () => {
    renderHistory();
    await waitFor(() => {
      expect(screen.getByText(/Game History/i)).toBeInTheDocument();
    });
  });

  it('renders rows when /games/history returns games', async () => {
    server.use(
      http.get('http://localhost:8000/games/history', () =>
        HttpResponse.json([
          {
            id: 7,
            status: 'COMPLETED',
            winner: 1,
            history: [],
            created_at: new Date().toISOString(),
            player_1_type: 'gpt-5',
            player_2_type: 'claude-3',
          },
        ])
      )
    );
    renderHistory();
    await waitFor(() => {
      expect(screen.getByText(/gpt-5/)).toBeInTheDocument();
      expect(screen.getByText(/claude-3/)).toBeInTheDocument();
    });
  });
});
