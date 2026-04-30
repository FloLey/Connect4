import { describe, expect, it, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';

import Tournament from '../../pages/Tournament';
import { DatabaseProvider } from '../../context/DatabaseContext';
import { ThemeProvider } from '../../context/ThemeContext';
import { ToastProvider } from '../../context/ToastContext';
import { recordedRequests } from '../../test/handlers';

const renderTournament = () =>
  render(
    <ToastProvider>
      <DatabaseProvider>
        <ThemeProvider>
          <Tournament />
        </ThemeProvider>
      </DatabaseProvider>
    </ToastProvider>
  );

beforeEach(() => {
  recordedRequests.reset();
});

describe('Tournament page', () => {
  it('renders the setup form when no tournament is active', async () => {
    renderTournament();

    // Setup screen mentions "Tournament Setup" once and references Round Robin.
    expect(await screen.findByText(/Tournament Setup/i)).toBeInTheDocument();
    const roundRobinHits = await screen.findAllByText(/Round Robin/i);
    expect(roundRobinHits.length).toBeGreaterThan(0);
    // And rounds / concurrency controls show up.
    expect(await screen.findByText(/Concurrency/i)).toBeInTheDocument();
  });
});
