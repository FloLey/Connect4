import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import PlayerCard from '../../../components/arena/PlayerCard';

describe('PlayerCard', () => {
  it('renders the player name', () => {
    render(<PlayerCard name="Alpha" type="ai" isActive color="red" />);
    expect(screen.getByText('Alpha')).toBeInTheDocument();
  });

  it('uses red styling for color="red"', () => {
    const { container } = render(
      <PlayerCard name="A" type="ai" isActive color="red" />
    );
    const dot = container.querySelector('.bg-red-500');
    expect(dot).not.toBeNull();
  });

  it('uses yellow styling for color="yellow"', () => {
    const { container } = render(
      <PlayerCard name="B" type="ai" isActive color="yellow" />
    );
    const dot = container.querySelector('.bg-yellow-400');
    expect(dot).not.toBeNull();
  });

  it('falls back to red when color is unknown', () => {
    const { container } = render(
      <PlayerCard name="X" type="ai" isActive color="purple" />
    );
    expect(container.querySelector('.bg-red-500')).not.toBeNull();
  });

  it('dims the card when not active', () => {
    const { container } = render(
      <PlayerCard name="A" type="ai" isActive={false} color="red" />
    );
    expect(container.querySelector('.opacity-60')).not.toBeNull();
  });
});
