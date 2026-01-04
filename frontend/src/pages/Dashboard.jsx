import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getLeaderboard, getActiveGames } from '../api/client';
import { Trophy, Activity, ArrowRight, Hash } from 'lucide-react';
import { useDatabase } from '../context/DatabaseContext';

const Dashboard = () => {
  const { dbEnv } = useDatabase();
  const [leaderboard, setLeaderboard] = useState([]);
  const [activeGames, setActiveGames] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    // Optional: Clear data immediately for visual feedback
    setLeaderboard([]);
    setActiveGames([]);

    const fetchData = async () => {
      try {
        const [lbData, gamesData] = await Promise.all([
          getLeaderboard(),
          getActiveGames()
        ]);
        setLeaderboard(lbData);
        setActiveGames(gamesData);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [dbEnv]);

  const Card = ({ title, icon: Icon, children, className = "" }) => (
    <div className={`bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm overflow-hidden flex flex-col ${className}`}>
      <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg text-brand-600 dark:text-brand-500">
            <Icon size={18} />
          </div>
          <h2 className="font-semibold text-gray-900 dark:text-white">{title}</h2>
        </div>
      </div>
      {children}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">System Overview</h1>
          <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">Real-time benchmarks and live games.</p>
        </div>
        <Link
          to="/new"
          className="inline-flex justify-center items-center px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white text-sm font-medium rounded-lg transition-colors shadow-sm"
        >
          Start New Match
        </Link>
      </div>

      {/* 
         FIX: Removed 'items-start'. 
         This restores default 'stretch' behavior so both cards in the row will match the height of the tallest one.
      */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Expanded Leaderboard */}
        <Card title="Model Performance" icon={Trophy} className="lg:col-span-3">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="bg-gray-50 dark:bg-gray-800/50 text-gray-500 dark:text-gray-400 uppercase text-xs">
                <tr>
                  <th className="px-6 py-3 font-medium">Rank</th>
                  <th className="px-6 py-3 font-medium">Model</th>
                  <th className="px-6 py-3 font-medium">ELO</th>
                  <th className="px-6 py-3 font-medium text-right">Win Rate</th>
                  <th className="px-6 py-3 font-medium text-right">Games Played</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                {leaderboard.map((entry, index) => {
                  const total = entry.wins + entry.losses + entry.draws;
                  const winRate = total > 0 ? ((entry.wins / total) * 100).toFixed(1) : 0;
                  return (
                    <tr key={entry.model_name} className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                      <td className="px-6 py-4 font-mono text-gray-400">#{index + 1}</td>
                      <td className="px-6 py-4 font-medium text-gray-900 dark:text-white">{entry.model_name}</td>
                      <td className="px-6 py-4 font-mono text-brand-600 dark:text-brand-400">{Math.round(entry.rating)}</td>
                      <td className="px-6 py-4 text-right">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                          winRate >= 50 
                            ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400' 
                            : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
                        }`}>
                          {winRate}%
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right text-gray-600 dark:text-gray-300">
                        {entry.matches_played}
                      </td>
                    </tr>
                  );
                })}
                {leaderboard.length === 0 && (
                  <tr><td colSpan="5" className="px-6 py-8 text-center text-gray-500">No data available</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Active Matches (Sidebar) */}
        <Card title="Live Activity" icon={Activity} className="lg:col-span-1 h-full min-h-[500px]">
          <div className="flex-1 min-h-0 overflow-y-auto divide-y divide-gray-100 dark:divide-gray-800">
            {activeGames.length === 0 ? (
              <div className="p-8 text-center text-gray-500 text-sm flex flex-col items-center justify-center h-full">
                No matches in progress.
              </div>
            ) : (
              activeGames.map((game) => (
                <Link
                  key={game.id}
                  to={`/game/${game.id}`}
                  className="block p-4 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors group"
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-mono text-gray-400">#{game.id}</span>
                    <div className="flex gap-1.5">
                       <span className="flex items-center gap-1 text-[10px] font-bold text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded-full">
                         <Hash size={9} /> {game.move_count}
                       </span>
                       <span className="flex items-center gap-1 text-[10px] font-bold tracking-wider text-brand-600 dark:text-brand-400 uppercase bg-brand-50 dark:bg-brand-900/20 px-1.5 py-0.5 rounded-full">
                        Live
                       </span>
                    </div>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                       <span className="text-gray-900 dark:text-gray-200 truncate text-xs font-medium">{game.player_1}</span>
                       <span className={`w-2 h-2 rounded-full bg-red-400 shadow-sm`}></span>
                    </div>
                    <div className="flex items-center justify-between">
                       <span className="text-gray-900 dark:text-gray-200 truncate text-xs font-medium">{game.player_2}</span>
                       <span className={`w-2 h-2 rounded-full bg-yellow-400 shadow-sm`}></span>
                    </div>
                  </div>
                  <div className="mt-3 flex items-center justify-end gap-1 text-[11px] text-brand-600 dark:text-brand-400 font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                    Watch <ArrowRight size={12} />
                  </div>
                </Link>
              ))
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;