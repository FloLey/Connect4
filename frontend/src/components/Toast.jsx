import { createPortal } from 'react-dom';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';
import { clsx } from 'clsx';

import { useToast, useToastQueue } from '../context/ToastContext';

const ICON_BY_LEVEL = {
  success: CheckCircle2,
  error: AlertCircle,
  info: Info,
};

const COLOR_BY_LEVEL = {
  success: 'border-green-500/40 bg-green-50 dark:bg-green-950/30 text-green-900 dark:text-green-200',
  error: 'border-red-500/40 bg-red-50 dark:bg-red-950/30 text-red-900 dark:text-red-200',
  info: 'border-brand-500/40 bg-brand-50 dark:bg-brand-950/30 text-brand-900 dark:text-brand-200',
};

const Toast = () => {
  const toasts = useToastQueue();
  const { dismiss } = useToast();

  if (typeof document === 'undefined') return null;

  return createPortal(
    <div
      role="region"
      aria-label="Notifications"
      className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none"
    >
      {toasts.map((t) => {
        const Icon = ICON_BY_LEVEL[t.level] ?? Info;
        return (
          <div
            key={t.id}
            data-testid="toast"
            data-level={t.level}
            role="alert"
            className={clsx(
              'flex items-start gap-3 p-3 border rounded-lg shadow-lg pointer-events-auto',
              COLOR_BY_LEVEL[t.level] ?? COLOR_BY_LEVEL.info
            )}
          >
            <Icon size={18} className="shrink-0 mt-0.5" />
            <span className="flex-1 text-sm">{t.message}</span>
            <button
              type="button"
              aria-label="Dismiss notification"
              onClick={() => dismiss(t.id)}
              className="shrink-0 -m-1 p-1 rounded hover:bg-black/5 dark:hover:bg-white/10"
            >
              <X size={14} />
            </button>
          </div>
        );
      })}
    </div>,
    document.body
  );
};

export default Toast;
