import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight as ArrowIcon, Clock, RefreshCw } from 'lucide-react';

import MiniGameBoard from '../../components/MiniGameBoard';
import { useGameSocket } from '../../hooks/useGameSocket';

const isSnoozed = (game) =>
  game.status === 'PAUSED' && game.retry_after && new Date(game.retry_after) > new Date();

/**
 * One card in the LiveGamesGrid. Subscribes to its own game's WebSocket so the
 * board + move count update in real time, independent of the tournament-level
 * polling cadence.
 *
 * The polled ``game`` snapshot still drives the static metadata (player names,
 * status, retry_after for snooze pills) and seeds the initial board until the
 * socket's first UPDATE arrives.
 */
const LiveGameCard = ({ game, tournamentStatus }) => {
  const { gameState } = useGameSocket(game.id);

  const board = gameState?.board ?? game.board;
  // Live move count comes from the WS-derived board (count non-zero cells)
  // since the UPDATE payload doesn't include move_count directly.
  const moveCount = useMemo(() => {
    if (!gameState?.board) return game.move_count || 0;
    let n = 0;
    for (const row of gameState.board) {
      for (const cell of row) {
        if (cell !== 0) n += 1;
      }
    }
    return n;
  }, [gameState?.board, game.move_count]);

  // WS-driven status takes precedence (e.g. game just completed) but the
  // snapshot's `retry_after` still controls the Snoozed pill.
  const liveStatus = gameState?.status ?? game.status;
  const snoozed = isSnoozed({ ...game, status: liveStatus });

  return (
    <Link
      to={`/game/${game.id}`}
      className="block bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-4 hover:border-brand-500 dark:hover:border-brand-500 hover:shadow-md transition-all group"
    >
      <div className="flex justify-between items-center mb-3">
        <span className="text-xs font-mono text-gray-400">#{game.id}</span>
        <div className="flex gap-2">
          {snoozed && (
            <span className="flex items-center gap-1 text-[10px] bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full animate-pulse">
              <Clock size={10} /> Rate Limit Cooldown
            </span>
          )}
          <span className="flex items-center gap-1 text-[10px] font-bold text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded-full">
            <RefreshCw
              size={10}
              className={tournamentStatus === 'IN_PROGRESS' ? 'animate-spin' : ''}
            />
            {moveCount}
          </span>
        </div>
      </div>

      <div className="flex gap-4">
        <div className="shrink-0">
          <MiniGameBoard board={board} />
        </div>
        <div className="flex-1 flex flex-col justify-center gap-2 overflow-hidden">
          <div className="flex items-center gap-2 min-w-0">
            <div className="w-2 h-2 rounded-full bg-red-500 shrink-0 shadow-sm" />
            <span
              className="text-xs font-medium text-gray-700 dark:text-gray-300 truncate"
              title={game.player_1}
            >
              {game.player_1}
            </span>
          </div>
          <div className="flex items-center gap-2 min-w-0">
            <div className="w-2 h-2 rounded-full bg-yellow-400 shrink-0 shadow-sm" />
            <span
              className="text-xs font-medium text-gray-700 dark:text-gray-300 truncate"
              title={game.player_2}
            >
              {game.player_2}
            </span>
          </div>
        </div>
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-gray-50 dark:border-gray-800 pt-2">
        {snoozed ? (
          <span className="flex items-center gap-1.5 text-[10px] font-bold tracking-wider text-amber-600 dark:text-amber-400 uppercase">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
            Snoozed
          </span>
        ) : (
          <span className="flex items-center gap-1.5 text-[10px] font-bold tracking-wider text-green-600 dark:text-green-400 uppercase">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            Live
          </span>
        )}
        <div className="text-gray-400 group-hover:text-brand-600 dark:group-hover:text-brand-400 transition-colors">
          <ArrowIcon size={14} />
        </div>
      </div>
    </Link>
  );
};

export default LiveGameCard;
