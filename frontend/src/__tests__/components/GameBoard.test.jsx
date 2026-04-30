import { describe, expect, it, vi } from 'vitest';
import { render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import GameBoard from '../../components/GameBoard';

const emptyBoard = () =>
  Array.from({ length: 6 }, () => Array.from({ length: 7 }, () => 0));

describe('GameBoard', () => {
  it('returns null when no board is provided', () => {
    const { container } = render(
      <GameBoard board={null} onColumnClick={() => {}} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders 42 cells (6 rows × 7 columns)', () => {
    const { container } = render(
      <GameBoard board={emptyBoard()} onColumnClick={() => {}} currentTurn={1} />
    );
    // Each cell is a div with the rounded-full hole class.
    const cells = container.querySelectorAll('.rounded-full.flex');
    expect(cells.length).toBe(42);
  });

  it('calls onColumnClick with the column index when a hole is clicked and it is the human turn', async () => {
    const onClick = vi.fn();
    const { container } = render(
      <GameBoard
        board={emptyBoard()}
        onColumnClick={onClick}
        isHumanTurn
        currentTurn={1}
      />
    );

    const user = userEvent.setup();
    // First clickable hole = top-left, column 0.
    const cells = container.querySelectorAll('.rounded-full.flex');
    await user.click(cells[0]);

    expect(onClick).toHaveBeenCalledWith(0);
  });

  it('does NOT call onColumnClick when there is a winner', async () => {
    const onClick = vi.fn();
    const { container } = render(
      <GameBoard
        board={emptyBoard()}
        onColumnClick={onClick}
        isHumanTurn
        currentTurn={1}
        winner={1}
      />
    );

    const user = userEvent.setup();
    const cells = container.querySelectorAll('.rounded-full.flex');
    await user.click(cells[0]);

    expect(onClick).not.toHaveBeenCalled();
  });

  it('does NOT call onColumnClick when isHumanTurn is false', async () => {
    const onClick = vi.fn();
    const { container } = render(
      <GameBoard
        board={emptyBoard()}
        onColumnClick={onClick}
        isHumanTurn={false}
        currentTurn={1}
      />
    );

    const user = userEvent.setup();
    const cells = container.querySelectorAll('.rounded-full.flex');
    await user.click(cells[0]);

    expect(onClick).not.toHaveBeenCalled();
  });
});
