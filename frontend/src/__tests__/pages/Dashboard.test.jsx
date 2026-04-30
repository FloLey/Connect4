import { describe, expect, it } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import Dashboard from '../../pages/Dashboard';
import { AllProviders } from '../../test/wrappers';

const renderDashboard = () =>
  render(
    <AllProviders>
      <MemoryRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Dashboard />
      </MemoryRouter>
    </AllProviders>
  );

describe('Dashboard page', () => {
  it('renders header and the empty-state messages from MSW defaults', async () => {
    renderDashboard();
    expect(screen.getByText(/System Overview/i)).toBeInTheDocument();
    expect(screen.getByText(/Start New Match/i)).toBeInTheDocument();

    // MSW returns empty arrays for both, so we render the placeholders.
    await waitFor(() => {
      expect(screen.getByText(/No data available/i)).toBeInTheDocument();
      expect(screen.getByText(/No matches in progress/i)).toBeInTheDocument();
    });
  });
});
