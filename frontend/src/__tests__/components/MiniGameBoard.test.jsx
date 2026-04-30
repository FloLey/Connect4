import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/react';
import MiniGameBoard from '../../components/MiniGameBoard';

const emptyBoard = () =>
  Array.from({ length: 6 }, () => Array.from({ length: 7 }, () => 0));

describe('MiniGameBoard', () => {
  it('returns null when board is missing', () => {
    const { container } = render(<MiniGameBoard board={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders 42 cells for a 6×7 board', () => {
    const { container } = render(<MiniGameBoard board={emptyBoard()} />);
    const cells = container.querySelectorAll('.rounded-full');
    expect(cells.length).toBe(42);
  });

  it('renders red for player 1 and yellow for player 2', () => {
    const board = emptyBoard();
    board[5][0] = 1;
    board[5][1] = 2;
    const { container } = render(<MiniGameBoard board={board} />);
    const cells = Array.from(container.querySelectorAll('.rounded-full'));
    // Bottom-row cells are at indices 35 and 36 (5*7 + 0/1) given top-to-bottom render order.
    expect(cells[35].className).toContain('bg-red-500');
    expect(cells[36].className).toContain('bg-yellow-400');
  });
});
