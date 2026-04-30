import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import TournamentSetupForm from '../../../pages/Tournament/TournamentSetupForm';

const models = [
  { id: 'a', label: 'Alpha' },
  { id: 'b', label: 'Beta' },
  { id: 'c', label: 'Gamma' },
];

describe('TournamentSetupForm', () => {
  it('does not render the target selector in Round Robin mode', () => {
    render(<TournamentSetupForm availableModels={models} onCreate={() => {}} />);
    expect(
      screen.queryByText(/Target Model \(Being Evaluated\)/i)
    ).not.toBeInTheDocument();
  });

  it('shows the target selector after switching to Evaluation mode', async () => {
    render(<TournamentSetupForm availableModels={models} onCreate={() => {}} />);
    await userEvent.click(screen.getByRole('button', { name: /Model Evaluation/i }));
    expect(
      await screen.findByText(/Target Model \(Being Evaluated\)/i)
    ).toBeInTheDocument();
  });

  it('disables Create when fewer than 2 models are selected', async () => {
    render(<TournamentSetupForm availableModels={[models[0]]} onCreate={() => {}} />);
    expect(screen.getByText(/Select at least 2 models/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Create/i })).not.toBeInTheDocument();
  });

  it('passes the chosen config to onCreate when clicking Create', async () => {
    const onCreate = vi.fn();
    render(<TournamentSetupForm availableModels={models} onCreate={onCreate} />);

    // Default selection is the first 3 models — Create button should be present.
    const createBtn = await screen.findByRole('button', { name: /Create/i });
    await userEvent.click(createBtn);

    expect(onCreate).toHaveBeenCalledWith({
      mode: 'ROUND_ROBIN',
      models: ['a', 'b', 'c'],
      rounds: 1,
      concurrency: 2,
      targetModel: '',
    });
  });

  it('blocks evaluation create until a target is selected', async () => {
    const onCreate = vi.fn();
    render(<TournamentSetupForm availableModels={models} onCreate={onCreate} />);
    await userEvent.click(screen.getByRole('button', { name: /Model Evaluation/i }));

    // No target picked → only the warning is rendered, not the Create button.
    expect(
      await screen.findByText(/Select a target model for evaluation/i)
    ).toBeInTheDocument();
    expect(onCreate).not.toHaveBeenCalled();

    const targetSection = screen
      .getByText(/Target Model \(Being Evaluated\)/i)
      .parentElement;
    await userEvent.click(within(targetSection).getByText('Alpha'));

    const createBtn = await screen.findByRole('button', { name: /Create/i });
    await userEvent.click(createBtn);
    expect(onCreate).toHaveBeenCalledWith(
      expect.objectContaining({ mode: 'EVALUATION', targetModel: 'a' })
    );
  });
});
