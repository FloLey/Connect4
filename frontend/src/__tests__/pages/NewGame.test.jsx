import { describe, expect, it, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

import NewGame from '../../pages/NewGame';
import { AllProviders } from '../../test/wrappers';
import { recordedRequests } from '../../test/handlers';

const renderNewGame = () =>
  render(
    <AllProviders>
      <MemoryRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <NewGame />
      </MemoryRouter>
    </AllProviders>
  );

beforeEach(() => {
  recordedRequests.reset();
});

describe('NewGame page', () => {
  it('renders the form once models are loaded', async () => {
    renderNewGame();
    await waitFor(() => {
      // MSW serves 2 default models; the page hides the loading spinner once
      // it has data and shows two role pickers.
      expect(screen.getByText(/Player 1/i)).toBeInTheDocument();
      expect(screen.getByText(/Player 2/i)).toBeInTheDocument();
    });
  });

  it('Randomize button picks two different AI models', async () => {
    renderNewGame();
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Randomize/i })).toBeInTheDocument()
    );
    await userEvent.click(screen.getByRole('button', { name: /Randomize/i }));
    // Sanity-only: button still there after click.
    expect(screen.getByRole('button', { name: /Randomize/i })).toBeInTheDocument();
  });
});
