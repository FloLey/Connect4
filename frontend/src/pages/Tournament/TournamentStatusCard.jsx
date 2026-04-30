import { useEffect, useState } from 'react';

const TournamentStatusCard = ({ tournament, onUpdateConcurrency }) => {
  const config = tournament.config || {};
  const [editingConcurrency, setEditingConcurrency] = useState(config.concurrency || 2);
  const [isSaving, setIsSaving] = useState(false);

  // Keep the editing slider in sync if the polled config changes.
  useEffect(() => {
    setEditingConcurrency(config.concurrency || 2);
  }, [config.concurrency]);

  const pct = Math.round((tournament.completed / tournament.total) * 100) || 0;

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await onUpdateConcurrency(editingConcurrency);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 shadow-sm">
      <div className="flex justify-between text-sm mb-2 text-gray-500">
        <span>Progress</span>
        <span>
          {tournament.completed} / {tournament.total} Games
        </span>
      </div>
      <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-4 mb-6 overflow-hidden">
        <div
          className="bg-brand-600 h-4 rounded-full transition-all duration-500 relative"
          style={{ width: `${pct}%` }}
        >
          <div className="absolute inset-0 bg-white/20 animate-[shimmer_2s_infinite]" />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className="text-gray-500">Status</div>
          <div className="font-bold dark:text-white">{tournament.status}</div>
        </div>
        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className="text-gray-500">Concurrency</div>
          {tournament.status === 'PAUSED' ? (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-bold dark:text-white">{editingConcurrency} Workers</span>
                <button
                  onClick={handleSave}
                  disabled={isSaving}
                  className="px-2 py-1 bg-brand-600 text-white text-xs rounded hover:bg-brand-700 disabled:opacity-50"
                >
                  {isSaving ? 'Saving…' : 'Save'}
                </button>
              </div>
              <input
                type="range"
                min="1"
                max="100"
                step="1"
                value={editingConcurrency}
                onChange={(e) => setEditingConcurrency(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-brand-600"
              />
              <p className="text-xs text-gray-500">Adjust while paused</p>
            </div>
          ) : (
            <div className="font-bold dark:text-white">{config.concurrency || 2} Workers</div>
          )}
        </div>
        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className="text-gray-500">Rounds</div>
          <div className="font-bold dark:text-white">{config.rounds || 1}</div>
        </div>
        <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <div className="text-gray-500">Models</div>
          <div className="font-bold dark:text-white">{config.model_ids?.length || 0}</div>
        </div>
      </div>
    </div>
  );
};

export default TournamentStatusCard;
