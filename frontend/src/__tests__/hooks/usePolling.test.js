import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

import { usePolling } from '../../hooks/usePolling';

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('usePolling', () => {
  it('fires once immediately, then every interval', () => {
    const fn = vi.fn();
    renderHook(() => usePolling(fn, 1000));

    expect(fn).toHaveBeenCalledTimes(1);

    act(() => vi.advanceTimersByTime(3000));
    expect(fn).toHaveBeenCalledTimes(4); // immediate + 3 ticks
  });

  it('clears the interval on unmount', () => {
    const fn = vi.fn();
    const { unmount } = renderHook(() => usePolling(fn, 500));

    expect(fn).toHaveBeenCalledTimes(1);
    unmount();
    act(() => vi.advanceTimersByTime(5000));
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('does not call fn when enabled is false', () => {
    const fn = vi.fn();
    renderHook(() => usePolling(fn, 1000, { enabled: false }));

    act(() => vi.advanceTimersByTime(5000));
    expect(fn).not.toHaveBeenCalled();
  });

  it('restarts when enabled flips back to true', () => {
    const fn = vi.fn();
    const { rerender } = renderHook(
      ({ enabled }) => usePolling(fn, 1000, { enabled }),
      { initialProps: { enabled: false } }
    );

    expect(fn).not.toHaveBeenCalled();
    rerender({ enabled: true });
    expect(fn).toHaveBeenCalledTimes(1);

    act(() => vi.advanceTimersByTime(2000));
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('restarts when interval changes', () => {
    const fn = vi.fn();
    const { rerender } = renderHook(({ interval }) => usePolling(fn, interval), {
      initialProps: { interval: 1000 },
    });

    expect(fn).toHaveBeenCalledTimes(1);
    rerender({ interval: 500 });
    expect(fn).toHaveBeenCalledTimes(2); // immediate again on restart

    act(() => vi.advanceTimersByTime(1000));
    expect(fn).toHaveBeenCalledTimes(4);
  });

  it('keeps polling when fn rejects', () => {
    const fn = vi.fn().mockRejectedValue(new Error('boom'));
    renderHook(() => usePolling(fn, 1000));

    act(() => vi.advanceTimersByTime(3000));
    expect(fn).toHaveBeenCalledTimes(4);
  });

  it('reads the latest fn each tick (no stale closure)', () => {
    let counter = 0;
    const { rerender } = renderHook(
      ({ value }) =>
        usePolling(() => {
          counter += value;
        }, 1000),
      { initialProps: { value: 1 } }
    );

    expect(counter).toBe(1);
    rerender({ value: 10 });
    act(() => vi.advanceTimersByTime(1000));
    expect(counter).toBe(11);
  });
});
