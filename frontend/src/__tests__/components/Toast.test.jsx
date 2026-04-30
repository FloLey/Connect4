import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ToastProvider, useToast } from '../../context/ToastContext';
import Toast from '../../components/Toast';

const Trigger = ({ message, level = 'success' }) => {
  const api = useToast();
  return (
    <button type="button" onClick={() => api[level](message, { durationMs: 0 })}>
      go
    </button>
  );
};

describe('Toast', () => {
  it('renders queued toasts via portal with the right level', async () => {
    render(
      <ToastProvider>
        <Trigger message="all good" level="success" />
        <Toast />
      </ToastProvider>
    );

    await userEvent.click(screen.getByRole('button', { name: 'go' }));

    const toast = await screen.findByTestId('toast');
    expect(toast).toHaveAttribute('data-level', 'success');
    expect(toast).toHaveTextContent('all good');
  });

  it('clicking the dismiss button removes the toast', async () => {
    render(
      <ToastProvider>
        <Trigger message="bye" level="info" />
        <Toast />
      </ToastProvider>
    );

    await userEvent.click(screen.getByRole('button', { name: 'go' }));
    expect(await screen.findByTestId('toast')).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole('button', { name: /Dismiss notification/i })
    );
    expect(screen.queryByTestId('toast')).not.toBeInTheDocument();
  });
});
