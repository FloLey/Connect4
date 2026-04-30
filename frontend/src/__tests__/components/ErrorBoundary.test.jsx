import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

import ErrorBoundary from '../../components/ErrorBoundary';

const Boom = () => {
  throw new Error('intentional test error');
};

const Safe = () => <div>safe content</div>;

let consoleSpy;
beforeEach(() => {
  consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
});
afterEach(() => {
  consoleSpy.mockRestore();
});

const renderWithRouter = (ui) =>
  render(
    <MemoryRouter
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      {ui}
    </MemoryRouter>
  );

describe('ErrorBoundary', () => {
  it('renders children when nothing throws', () => {
    renderWithRouter(
      <ErrorBoundary>
        <Safe />
      </ErrorBoundary>
    );
    expect(screen.getByText('safe content')).toBeInTheDocument();
  });

  it('shows fallback panel when a child throws', () => {
    renderWithRouter(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    );
    expect(screen.getByText(/Something went wrong/i)).toBeInTheDocument();
    expect(screen.getByText(/intentional test error/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Reload page/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Back to Home/i })).toBeInTheDocument();
  });

  it('logs the original error to the console (so dev still sees it)', () => {
    renderWithRouter(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    );
    expect(consoleSpy).toHaveBeenCalled();
    const calls = consoleSpy.mock.calls.flat().map(String).join(' ');
    expect(calls).toContain('intentional test error');
  });

  it('reload button calls window.location.reload', async () => {
    const reloadSpy = vi.fn();
    const original = window.location;
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...original, reload: reloadSpy },
    });

    renderWithRouter(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    );

    await userEvent.click(screen.getByRole('button', { name: /Reload page/i }));
    expect(reloadSpy).toHaveBeenCalled();

    Object.defineProperty(window, 'location', {
      configurable: true,
      value: original,
    });
  });
});
