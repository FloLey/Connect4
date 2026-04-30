import { Play, Square, Trophy } from 'lucide-react';

import Loading from '../../components/Loading';
import { useTournamentState } from '../../hooks/useTournamentState';
import TournamentSetupForm from './TournamentSetupForm';
import TournamentStatusCard from './TournamentStatusCard';
import LiveGamesGrid from './LiveGamesGrid';
import TournamentCompletionScreen from './TournamentCompletionScreen';

const Tournament = () => {
  const {
    phase,
    tournament,
    activeGames,
    availableModels,
    actions,
  } = useTournamentState();

  if (phase === 'loading') return <Loading />;

  if (phase === 'setup') {
    return (
      <TournamentSetupForm
        availableModels={availableModels}
        onCreate={actions.create}
      />
    );
  }

  if (phase === 'completion' && tournament) {
    return (
      <TournamentCompletionScreen
        tournament={tournament}
        onReset={actions.dismiss}
      />
    );
  }

  // 'live' phase — tournament is non-null per the reducer.
  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold flex items-center gap-2 dark:text-white">
          <Trophy className="text-yellow-500" /> Tournament #{tournament.id}
        </h1>
        <div className="flex gap-2">
          {tournament.status === 'SETUP' && (
            <button
              onClick={actions.start}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center gap-2"
            >
              <Play size={18} /> Start Now
            </button>
          )}
          {tournament.status === 'IN_PROGRESS' && (
            <>
              <button
                onClick={actions.pause}
                className="bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-lg flex items-center gap-2"
              >
                <Square size={18} /> Pause
              </button>
              <button
                onClick={actions.stop}
                className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg flex items-center gap-2"
              >
                <Square size={18} /> Stop
              </button>
            </>
          )}
          {tournament.status === 'PAUSED' && (
            <button
              onClick={actions.resume}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center gap-2"
            >
              <Play size={18} /> Resume
            </button>
          )}
        </div>
      </div>

      <TournamentStatusCard
        tournament={tournament}
        onUpdateConcurrency={actions.updateConcurrency}
      />

      <LiveGamesGrid
        tournamentStatus={tournament.status}
        activeGames={activeGames}
      />
    </div>
  );
};

export default Tournament;
