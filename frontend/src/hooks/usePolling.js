import { useEffect, useRef } from 'react';

/**
 * Periodically invoke ``fn``. Calls ``fn`` once immediately, then again every
 * ``intervalMs`` while ``enabled`` is true. ``deps`` propagates into the
 * underlying useEffect so callers can re-trigger on context changes (e.g.
 * dbEnv switch).
 *
 * Stops the interval cleanly on unmount or when ``enabled`` flips to false.
 * Uses a ref for the latest ``fn`` so callers don't have to memoize it.
 */
export const usePolling = (fn, intervalMs, { enabled = true, deps = [] } = {}) => {
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (!enabled || !intervalMs || intervalMs <= 0) return undefined;

    let cancelled = false;
    const tick = () => {
      if (cancelled) return;
      const result = fnRef.current?.();
      // Swallow rejection so a single failure doesn't kill the interval.
      if (result && typeof result.then === 'function') {
        result.catch(() => {});
      }
    };

    tick();
    const id = setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, intervalMs, ...deps]);
};
