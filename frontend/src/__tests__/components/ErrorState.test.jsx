import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import ErrorState from '../../components/ErrorState';

describe('ErrorState', () => {
  it('renders the message', () => {
    render(<ErrorState message="something broke" />);
    expect(screen.getByText('something broke')).toBeInTheDocument();
  });

  it('renders the default message when none is provided', () => {
    render(<ErrorState />);
    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
  });

  it('does not render a Retry button when onRetry is not provided', () => {
    render(<ErrorState message="oops" />);
    expect(
      screen.queryByRole('button', { name: /Retry/i })
    ).not.toBeInTheDocument();
  });

  it('Retry button calls onRetry', async () => {
    const onRetry = vi.fn();
    render(<ErrorState message="oops" onRetry={onRetry} />);
    await userEvent.click(screen.getByRole('button', { name: /Retry/i }));
    expect(onRetry).toHaveBeenCalled();
  });
});
