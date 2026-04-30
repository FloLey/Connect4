import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import ReplayControls from '../../../components/arena/ReplayControls';

const renderControls = (overrides = {}) => {
  const props = {
    step: 2,
    total: 5,
    isPlaying: false,
    onSeek: vi.fn(),
    onPlayPause: vi.fn(),
    ...overrides,
  };
  return { ...render(<ReplayControls {...props} />), props };
};

describe('ReplayControls', () => {
  it('renders the move counter', () => {
    renderControls();
    expect(screen.getByText(/Move 3 \/ 5/)).toBeInTheDocument();
  });

  it('renders the range slider for scrubbing', () => {
    renderControls();
    expect(screen.getByRole('slider')).toBeInTheDocument();
  });

  it('Play button calls onPlayPause', async () => {
    const { props, container } = renderControls();
    const buttons = container.querySelectorAll('button');
    // The middle button (index 1) is the play/pause button.
    await userEvent.click(buttons[1]);
    expect(props.onPlayPause).toHaveBeenCalled();
  });

  it('SkipBack button seeks to 0', async () => {
    const { props, container } = renderControls();
    const buttons = container.querySelectorAll('button');
    await userEvent.click(buttons[0]);
    expect(props.onSeek).toHaveBeenCalledWith(0);
  });

  it('SkipForward button seeks to total - 1', async () => {
    const { props, container } = renderControls();
    const buttons = container.querySelectorAll('button');
    await userEvent.click(buttons[2]);
    expect(props.onSeek).toHaveBeenCalledWith(4);
  });

  it('SkipBack disabled at step 0', () => {
    const { container } = renderControls({ step: 0 });
    const buttons = container.querySelectorAll('button');
    expect(buttons[0]).toBeDisabled();
  });

  it('Play and SkipForward disabled at the last step', () => {
    const { container } = renderControls({ step: 4 });
    const buttons = container.querySelectorAll('button');
    expect(buttons[1]).toBeDisabled();
    expect(buttons[2]).toBeDisabled();
  });
});
