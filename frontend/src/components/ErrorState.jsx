import { AlertCircle, RefreshCw } from 'lucide-react';
import { clsx } from 'clsx';

const ErrorState = ({ message, onRetry, className = '' }) => (
  <div
    className={clsx(
      'p-10 flex flex-col items-center justify-center gap-3 text-center',
      className
    )}
    role="alert"
  >
    <div className="p-2 bg-red-100 dark:bg-red-900/40 rounded-lg text-red-600 dark:text-red-400">
      <AlertCircle size={20} />
    </div>
    <p className="text-sm text-gray-700 dark:text-gray-300 max-w-md">
      {message || 'Something went wrong.'}
    </p>
    {onRetry && (
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800"
      >
        <RefreshCw size={14} /> Retry
      </button>
    )}
  </div>
);

export default ErrorState;
