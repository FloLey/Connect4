import { describe, expect, it, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import Admin from '../../pages/Admin';
import { AllProviders } from '../../test/wrappers';

const renderAdmin = () =>
  render(
    <AllProviders>
      <MemoryRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Admin />
      </MemoryRouter>
    </AllProviders>
  );

beforeEach(() => {
  localStorage.clear();
});

describe('Admin page', () => {
  it('renders the header and stats from the MSW handler', async () => {
    renderAdmin();
    await waitFor(() =>
      expect(screen.getByText('Administration')).toBeInTheDocument()
    );
    // MSW default handler returns games=0, elo_ratings=0, elo_history=0.
    await waitFor(() => {
      expect(screen.getAllByText('0').length).toBeGreaterThanOrEqual(3);
    });
  });

  it('exposes the "Clear admin token" button', async () => {
    renderAdmin();
    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: /Clear admin token/i })
      ).toBeInTheDocument()
    );
  });
});
