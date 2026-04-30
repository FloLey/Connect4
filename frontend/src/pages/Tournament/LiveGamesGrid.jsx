import { Activity } from 'lucide-react';

import LiveGameCard from './LiveGameCard';

const LiveGamesGrid = ({ tournamentStatus, activeGames }) => (
  <div className="space-y-4">
    <h2 className="text-lg font-semibold flex items-center gap-2 dark:text-white">
      <Activity size={20} className="text-brand-500" /> Live Matches
    </h2>

    {activeGames.length === 0 ? (
      <div className="p-8 text-center text-gray-500 bg-white dark:bg-gray-900 rounded-xl border border-dashed border-gray-300 dark:border-gray-700">
        {tournamentStatus === 'SETUP' ? 'Waiting to start…' : 'Spinning up workers…'}
      </div>
    ) : (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {activeGames.map((game) => (
          <LiveGameCard
            key={game.id}
            game={game}
            tournamentStatus={tournamentStatus}
          />
        ))}
      </div>
    )}
  </div>
);

export default LiveGamesGrid;
