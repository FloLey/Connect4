import { describe, expect, it, beforeEach, vi, afterEach } from 'vitest';
import { render } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import React from 'react';

import Arena from '../../pages/Arena';
import { DatabaseProvider } from '../../context/DatabaseContext';
import { ThemeProvider } from '../../context/ThemeContext';
import { ToastProvider } from '../../context/ToastContext';

class MockWebSocket {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    MockWebSocket.instances.push(this);
  }
  send() {}
  close() {
    this.readyState = 3;
    this.onclose && this.onclose();
  }
}
MockWebSocket.OPEN = 1;

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.stubGlobal('WebSocket', MockWebSocket);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('Arena page', () => {
  it('mounts at /game/:id and opens a WebSocket for that game', async () => {
    render(
      <ToastProvider>
        <DatabaseProvider>
          <ThemeProvider>
            <MemoryRouter
              initialEntries={['/game/123']}
              future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
            >
              <Routes>
                <Route path="/game/:id" element={<Arena />} />
              </Routes>
            </MemoryRouter>
          </ThemeProvider>
        </DatabaseProvider>
      </ToastProvider>
    );

    // The hook creates a WebSocket immediately for the route param.
    expect(MockWebSocket.instances.length).toBeGreaterThanOrEqual(1);
    expect(MockWebSocket.instances[0].url).toContain('/games/123/ws');
  });
});
