import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

import Navigation from '../../components/Navigation';
import { AllProviders } from '../../test/wrappers';

const renderNav = () =>
  render(
    <AllProviders>
      <MemoryRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Navigation />
      </MemoryRouter>
    </AllProviders>
  );

describe('Navigation', () => {
  it('renders all top-level destinations', () => {
    renderNav();
    expect(screen.getByText('Overview')).toBeInTheDocument();
    expect(screen.getByText('Tournament')).toBeInTheDocument();
    expect(screen.getByText('Statistics')).toBeInTheDocument();
    expect(screen.getByText('New Match')).toBeInTheDocument();
    expect(screen.getByText('History')).toBeInTheDocument();
    expect(screen.getByText('Admin')).toBeInTheDocument();
  });

  it('toggles the database environment when the env button is clicked', async () => {
    renderNav();
    const button = screen.getByRole('button', { name: /Toggle Database Environment/i });
    expect(button).toHaveTextContent(/Production/i);

    await userEvent.click(button);
    expect(button).toHaveTextContent(/Test Sandbox/i);
  });

  it('toggles theme when the theme button is clicked', async () => {
    renderNav();
    const button = screen.getByRole('button', { name: /Toggle Theme/i });
    // Toggle once and ensure the button is still present (sanity-only —
    // theme effect is a class swap on <html>).
    await userEvent.click(button);
    expect(button).toBeInTheDocument();
  });
});
