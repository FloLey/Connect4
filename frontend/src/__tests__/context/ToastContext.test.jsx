import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

import { ToastProvider, useToast, useToastQueue } from '../../context/ToastContext';

const wrapper = ({ children }) => (
  <ToastProvider defaultDurationMs={3000}>{children}</ToastProvider>
);

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('ToastContext', () => {
  it('enqueues success/error/info with the right level', () => {
    const { result } = renderHook(
      () => ({ api: useToast(), queue: useToastQueue() }),
      { wrapper }
    );

    act(() => {
      result.current.api.success('Saved!');
      result.current.api.error('Boom');
      result.current.api.info('FYI');
    });

    expect(result.current.queue).toHaveLength(3);
    const levels = result.current.queue.map((t) => t.level);
    expect(levels).toEqual(['success', 'error', 'info']);
  });

  it('auto-dismisses after the configured duration', () => {
    const { result } = renderHook(
      () => ({ api: useToast(), queue: useToastQueue() }),
      { wrapper }
    );

    act(() => {
      result.current.api.success('temporary');
    });
    expect(result.current.queue).toHaveLength(1);

    act(() => {
      vi.advanceTimersByTime(3500);
    });
    expect(result.current.queue).toHaveLength(0);
  });

  it('manual dismiss removes the toast', () => {
    const { result } = renderHook(
      () => ({ api: useToast(), queue: useToastQueue() }),
      { wrapper }
    );

    let id;
    act(() => {
      id = result.current.api.error('persistent');
    });
    expect(result.current.queue).toHaveLength(1);

    act(() => {
      result.current.api.dismiss(id);
    });
    expect(result.current.queue).toHaveLength(0);
  });

  it('durationMs=0 keeps the toast until manually dismissed', () => {
    const { result } = renderHook(
      () => ({ api: useToast(), queue: useToastQueue() }),
      { wrapper }
    );

    act(() => {
      result.current.api.error('sticky', { durationMs: 0 });
    });

    act(() => vi.advanceTimersByTime(60000));
    expect(result.current.queue).toHaveLength(1);
  });

  it('throws if useToast is used outside the provider', () => {
    expect(() => renderHook(() => useToast())).toThrow(/ToastProvider/);
  });
});
