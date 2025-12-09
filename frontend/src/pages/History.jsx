import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { History as HistoryIcon, User, Cpu, Trophy, Play, Calendar, Hash, RefreshCw, Clock, Zap } from 'lucide-react';
import { getGameHistory } from '../api/client';

const History = () => {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);

  const GAMES_PER_PAGE = 50;

  useEffect(() => {
    // Initial load only
    loadGames(0, true);
  }, []);

  const loadGames = async (page, isReset) => {
    setLoading(true);
    try {
      const skip = page * GAMES_PER_PAGE;
      const data = await getGameHistory(skip, GAMES_PER_PAGE);
      
      if (isReset) {
        setGames(data);
      } else {
        setGames(prev => [...prev, ...data]);
      }
      setHasMore(data.length === GAMES_PER_PAGE);
    } catch (e) {
      console.error('Failed to fetch game history:', e);
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = () => {
    setCurrentPage(0);
    loadGames(0, true);
  };

  const onLoadMore = () => {
    const nextPage = currentPage + 1;
    setCurrentPage(nextPage);
    loadGames(nextPage, false);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getPlayerIcon = (playerType) => {
    return playerType === 'human' ? <User size={16} /> : <Cpu size={16} />;
  };

  const getResultDisplay = (game) => {
    if (game.status === 'DRAW') {
      return <span className="text-yellow-400">Draw</span>;
    }
    
    if (game.winner === 1) {
      return <span className="text-green-400">P1 Win</span>;
    } else if (game.winner === 2) {
      return <span className="text-blue-400">P2 Win</span>;
    }
    
    return <span className="text-slate-400">Unknown</span>;
  };

  const getPlayerName = (playerType) => {
    return playerType === 'human' ? 'Human' : playerType;
  };

  // Helper to sum up stats from history array
  const getGameStats = (history) => {
    if (!history) return { p1Time: 0, p2Time: 0, p1Tokens: 0, p2Tokens: 0 };
    
    let p1Time = 0, p2Time = 0, p1Tokens = 0, p2Tokens = 0;
    
    history.forEach(move => {
      if (move.player === 1) {
        p1Time += move.duration || 0;
        p1Tokens += move.output_tokens || 0;
      } else {
        p2Time += move.duration || 0;
        p2Tokens += move.output_tokens || 0;
      }
    });
    
    return { 
      p1Time: p1Time.toFixed(1), 
      p2Time: p2Time.toFixed(1), 
      p1Tokens, 
      p2Tokens 
    };
  };

  if (loading && games.length === 0) {
    return (
      <div className="flex justify-center items-center min-h-[400px]">
        <div className="text-slate-400">Loading game history...</div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <HistoryIcon className="text-brand-600 dark:text-brand-500" size={32} />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Game History</h1>
        </div>
        
        {/* Refresh Button */}
        <button 
          onClick={onRefresh}
          className="p-2 bg-white dark:bg-gray-900 hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-500 dark:text-gray-400 rounded-lg border border-gray-200 dark:border-gray-700 transition-colors"
          title="Refresh List"
        >
          <RefreshCw size={20} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {games.length === 0 ? (
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-8 text-center">
          <HistoryIcon className="mx-auto text-gray-400 dark:text-gray-500 mb-4" size={48} />
          <h2 className="text-xl text-gray-700 dark:text-gray-300 mb-2">No Games Found</h2>
          <p className="text-gray-500 dark:text-gray-400">No completed games to display yet.</p>
          <Link 
            to="/new" 
            className="mt-4 inline-block bg-brand-600 hover:bg-brand-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Start a New Game
          </Link>
        </div>
      ) : (
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden shadow-sm">
          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 text-xs uppercase">
                <tr>
                  <th className="px-4 py-3 text-left">Date</th>
                  <th className="px-4 py-3 text-left">ID</th>
                  <th className="px-4 py-3 text-left">Player 1 (Red)</th>
                  <th className="px-4 py-3 text-left">Player 2 (Yel)</th>
                  <th className="px-4 py-3 text-center">Result</th>
                  <th className="px-4 py-3 text-right">P1 Stats (Time / Tok)</th>
                  <th className="px-4 py-3 text-right">P2 Stats (Time / Tok)</th>
                  <th className="px-4 py-3 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800 text-sm">
                {games.map((game) => {
                  const stats = getGameStats(game.history);
                  return (
                    <tr key={game.id} className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                      <td className="px-4 py-3 text-gray-500">{formatDate(game.created_at)}</td>
                      <td className="px-4 py-3 font-mono text-brand-600">#{game.id}</td>
                      
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {getPlayerIcon(game.player_1_type)}
                          <span className={game.winner === 1 ? 'font-bold text-gray-900 dark:text-white' : 'text-gray-600 dark:text-gray-400'}>
                            {game.player_1_type}
                          </span>
                        </div>
                      </td>
                      
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {getPlayerIcon(game.player_2_type)}
                          <span className={game.winner === 2 ? 'font-bold text-gray-900 dark:text-white' : 'text-gray-600 dark:text-gray-400'}>
                            {game.player_2_type}
                          </span>
                        </div>
                      </td>

                      <td className="px-4 py-3 text-center font-bold">
                        {game.status === 'DRAW' ? (
                          <span className="text-yellow-500">Draw</span>
                        ) : game.winner === 1 ? (
                          <span className="text-green-500">P1 Win</span>
                        ) : (
                          <span className="text-blue-500">P2 Win</span>
                        )}
                      </td>

                      {/* Stats Columns */}
                      <td className="px-4 py-3 text-right font-mono text-xs text-gray-500">
                         {stats.p1Time}s / {stats.p1Tokens}T
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-xs text-gray-500">
                         {stats.p2Time}s / {stats.p2Tokens}T
                      </td>

                      <td className="px-4 py-3 text-center">
                        <Link to={`/game/${game.id}`} className="text-brand-600 hover:underline text-xs font-bold">
                          View
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {hasMore && (
            <div className="border-t border-gray-100 dark:border-gray-800 p-4 text-center">
              <button
                onClick={onLoadMore}
                disabled={loading}
                className="bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 disabled:bg-gray-50 dark:disabled:bg-gray-900 text-gray-700 dark:text-gray-200 disabled:text-gray-400 dark:disabled:text-gray-500 px-4 py-2 rounded-lg transition-colors disabled:cursor-not-allowed"
              >
                {loading ? 'Loading...' : 'Load More'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default History;