import { Loader2 } from 'lucide-react';
import { clsx } from 'clsx';

const Loading = ({ message = 'Loading…', className = '' }) => (
  <div
    className={clsx(
      'p-10 flex items-center justify-center gap-2 text-gray-500 dark:text-gray-400',
      className
    )}
    role="status"
    aria-live="polite"
  >
    <Loader2 size={18} className="animate-spin" />
    <span className="text-sm">{message}</span>
  </div>
);

export default Loading;
