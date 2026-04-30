import { createContext, useCallback, useContext, useMemo, useRef, useState } from 'react';

const ToastContext = createContext(null);

const DEFAULT_DURATION_MS = 4000;

let _seq = 0;
const nextId = () => {
  _seq += 1;
  return _seq;
};

export const ToastProvider = ({ children, defaultDurationMs = DEFAULT_DURATION_MS }) => {
  const [toasts, setToasts] = useState([]);
  const timersRef = useRef(new Map());

  const dismiss = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    const timer = timersRef.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timersRef.current.delete(id);
    }
  }, []);

  const enqueue = useCallback(
    (level, message, { durationMs = defaultDurationMs } = {}) => {
      const id = nextId();
      setToasts((prev) => [...prev, { id, level, message }]);
      if (durationMs > 0) {
        const timer = setTimeout(() => dismiss(id), durationMs);
        timersRef.current.set(id, timer);
      }
      return id;
    },
    [defaultDurationMs, dismiss]
  );

  const api = useMemo(
    () => ({
      success: (msg, opts) => enqueue('success', msg, opts),
      error: (msg, opts) => enqueue('error', msg, opts),
      info: (msg, opts) => enqueue('info', msg, opts),
      dismiss,
    }),
    [enqueue, dismiss]
  );

  return (
    <ToastContext.Provider value={{ toasts, ...api }}>
      {children}
    </ToastContext.Provider>
  );
};

export const useToast = () => {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error('useToast must be used within a <ToastProvider>');
  }
  return ctx;
};

export const useToastQueue = () => {
  const ctx = useContext(ToastContext);
  return ctx?.toasts ?? [];
};
