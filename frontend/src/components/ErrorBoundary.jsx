import React from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle } from 'lucide-react';

/**
 * Page-level error boundary. Wrap individual <Route element={...}> children so
 * one page crashing doesn't kill the navigation chrome.
 *
 * componentDidCatch also logs to the console so dev workflows still surface
 * the original error stack — the fallback UI is for users, the log is for us.
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, errorInfo) {
    // Surface the original error in dev tools too.
    // eslint-disable-next-line no-console
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({ error: null });
  };

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <div className="max-w-2xl mx-auto mt-12 p-8 bg-white dark:bg-gray-900 border border-red-200 dark:border-red-900 rounded-xl shadow-sm">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-red-100 dark:bg-red-900/40 rounded-lg text-red-600 dark:text-red-400">
            <AlertTriangle size={22} />
          </div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            Something went wrong on this page.
          </h2>
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
          The rest of the app is still working — use the navigation above to keep going,
          reload this page, or jump to the dashboard.
        </p>
        {this.state.error?.message && (
          <pre className="mt-4 mb-4 p-3 bg-gray-50 dark:bg-gray-800 text-xs text-red-700 dark:text-red-300 rounded overflow-x-auto">
            {String(this.state.error.message)}
          </pre>
        )}
        <div className="flex gap-2 mt-6">
          <button
            type="button"
            onClick={this.handleReload}
            className="px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white text-sm font-medium rounded-lg transition-colors"
          >
            Reload page
          </button>
          <Link
            to="/"
            onClick={this.handleReset}
            className="px-4 py-2 border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 text-sm font-medium rounded-lg transition-colors"
          >
            Back to Home
          </Link>
        </div>
      </div>
    );
  }
}

export default ErrorBoundary;
