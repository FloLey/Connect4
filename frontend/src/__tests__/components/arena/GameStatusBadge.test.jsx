import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import GameStatusBadge from '../../../components/arena/GameStatusBadge';

describe('GameStatusBadge', () => {
  it('renders the spectating variant', () => {
    render(<GameStatusBadge type="spectating" />);
    expect(screen.getByText(/Spectating/i)).toBeInTheDocument();
  });

  it('renders the replay variant', () => {
    render(<GameStatusBadge type="replay" />);
    expect(screen.getByText(/Replay/i)).toBeInTheDocument();
  });

  it('shows LIVE when connected', () => {
    render(<GameStatusBadge type="live" isConnected />);
    expect(screen.getByText('LIVE')).toBeInTheDocument();
  });

  it('shows OFFLINE when disconnected', () => {
    render(<GameStatusBadge type="live" isConnected={false} />);
    expect(screen.getByText('OFFLINE')).toBeInTheDocument();
  });

  it('returns null for unknown type', () => {
    const { container } = render(<GameStatusBadge type="???" />);
    expect(container.firstChild).toBeNull();
  });
});
