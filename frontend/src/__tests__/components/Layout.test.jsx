import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import Layout from '../../components/Layout';
import { AllProviders } from '../../test/wrappers';

const renderLayout = (children = <div>page content</div>) =>
  render(
    <AllProviders>
      <MemoryRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Layout>{children}</Layout>
      </MemoryRouter>
    </AllProviders>
  );

describe('Layout', () => {
  it('renders the navigation chrome and the children', () => {
    renderLayout();
    // Navigation logo is always there.
    expect(screen.getByText(/Connect/i)).toBeInTheDocument();
    expect(screen.getByText('page content')).toBeInTheDocument();
  });

  it('does NOT show the test-env banner by default', () => {
    renderLayout();
    expect(screen.queryByText(/TEST ENVIRONMENT/i)).not.toBeInTheDocument();
  });

  it('shows the test-env banner when dbEnv is "test"', () => {
    localStorage.setItem('dbEnv', 'test');
    try {
      renderLayout();
      expect(screen.getByText(/TEST ENVIRONMENT/i)).toBeInTheDocument();
    } finally {
      localStorage.removeItem('dbEnv');
    }
  });
});
