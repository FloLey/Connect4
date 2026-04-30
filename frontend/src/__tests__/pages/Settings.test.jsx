import { describe, expect, it, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

import Settings from '../../pages/Settings';
import { AllProviders } from '../../test/wrappers';
import { recordedRequests } from '../../test/handlers';

const renderSettings = () =>
  render(
    <AllProviders>
      <MemoryRouter
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Settings />
      </MemoryRouter>
    </AllProviders>
  );

beforeEach(() => {
  recordedRequests.reset();
});

describe('Settings page', () => {
  it('renders provider rows with masked keys', async () => {
    renderSettings();
    await waitFor(() => {
      expect(screen.getByText('OpenAI')).toBeInTheDocument();
    });
    // OpenAI is set via env (per MSW default).
    expect(screen.getByText('****ABCD')).toBeInTheDocument();
    // Mistral is overridden.
    expect(screen.getByText('****WXYZ')).toBeInTheDocument();
    // Anthropic / Google / DeepSeek are unset.
    const notSetCount = screen.getAllByText(/— not set —/i).length;
    expect(notSetCount).toBeGreaterThanOrEqual(3);
  });

  it('renders all editable tunables with their default values', async () => {
    renderSettings();
    await waitFor(() => {
      expect(screen.getByText(/ELO K-factor/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Default temperature/i)).toBeInTheDocument();
    expect(screen.getByText(/Fallback model/i)).toBeInTheDocument();
    expect(screen.getByText(/Rate-limit snooze/i)).toBeInTheDocument();
    expect(screen.getByText(/Game runner pacing/i)).toBeInTheDocument();
  });

  it('clicking the eye icon on a provider opens an inline editor', async () => {
    renderSettings();
    await waitFor(() => screen.getByText('OpenAI'));

    // 5 providers × 1 toggle each + clear buttons. Find the OpenAI row.
    const openaiRow = screen.getByText('OpenAI').closest('div').parentElement;
    const editBtn = openaiRow.querySelector(
      'button[aria-label="Edit"], button[aria-label="Cancel"]'
    );
    await userEvent.click(editBtn);

    expect(openaiRow.querySelector('input[type="text"]')).not.toBeNull();
  });

  it('Save button PATCHes the changed key', async () => {
    renderSettings();
    await waitFor(() => screen.getByText('OpenAI'));

    const openaiRow = screen.getByText('OpenAI').closest('div').parentElement;
    await userEvent.click(openaiRow.querySelector('button[aria-label="Edit"]'));
    const input = openaiRow.querySelector('input[type="text"]');
    await userEvent.type(input, 'sk-new-key');

    await userEvent.click(screen.getByRole('button', { name: /Save changes/i }));

    await waitFor(() => {
      const patches = recordedRequests.list.filter((r) => r.method === 'PATCH');
      expect(patches.length).toBeGreaterThan(0);
      expect(patches[0].body.api_keys).toEqual({ openai: 'sk-new-key' });
    });
  });

  it('reset button on an overridden tunable calls DELETE', async () => {
    // Override the MSW response so elo_k_factor shows as overridden.
    const { server } = await import('../../test/msw-server');
    const { http, HttpResponse } = await import('msw');
    server.use(
      http.get('http://localhost:8000/settings', () =>
        HttpResponse.json({
          providers: ['openai'],
          editable_tunables: ['elo_k_factor'],
          api_keys: { openai: { set: false, source: null, preview: null } },
          tunables: {
            elo_k_factor: { value: 24, default: 32, overridden: true },
          },
        })
      )
    );

    renderSettings();
    const resetBtn = await screen.findByRole('button', { name: /reset \(default 32\)/i });
    await userEvent.click(resetBtn);

    await waitFor(() => {
      const deletes = recordedRequests.list.filter((r) => r.method === 'DELETE');
      expect(deletes.some((r) => r.url.endsWith('/settings/tunables/elo_k_factor'))).toBe(true);
    });
  });
});
