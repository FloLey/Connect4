import { Link } from 'react-router-dom';
import { CheckCircle2 } from 'lucide-react';

const TournamentCompletionScreen = ({ tournament, onReset }) => (
  <div className="max-w-2xl mx-auto text-center space-y-6 pt-10">
    <div className="bg-white dark:bg-gray-900 p-8 rounded-2xl border border-gray-200 dark:border-gray-800 shadow-sm">
      <CheckCircle2 size={64} className="mx-auto text-green-500 mb-4" />
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Tournament Finished</h1>
      <p className="text-gray-500 mb-6">
        All {tournament.total} scheduled matches have been processed.
      </p>
      <div className="flex justify-center gap-4">
        <button
          onClick={onReset}
          className="px-6 py-2 bg-brand-600 text-white rounded-lg hover:bg-brand-700"
        >
          Start New Tournament
        </button>
        <Link
          to="/statistics"
          className="px-6 py-2 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white rounded-lg hover:bg-gray-200"
        >
          View Leaderboard
        </Link>
      </div>
    </div>
  </div>
);

export default TournamentCompletionScreen;
