import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertTriangle, Database, Trash2, RefreshCw, Shield } from 'lucide-react';
import { getAdminStatus, resetDatabase } from '../api/client';

const Admin = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);
  const [isResetting, setIsResetting] = useState(false);
  const [confirmText, setConfirmText] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const data = await getAdminStatus();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch admin stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleResetDatabase = async () => {
    if (confirmText !== 'DELETE') {
      alert('You must type "DELETE" to confirm');
      return;
    }

    setIsResetting(true);
    try {
      await resetDatabase();
      alert('Database successfully reset!');
      setShowModal(false);
      setConfirmText('');
      await fetchStats(); // Refresh stats
      // Optionally redirect to dashboard
      navigate('/');
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Network error';
      alert(`Reset failed: ${errorMessage}`);
      console.error('Reset error:', error);
    } finally {
      setIsResetting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-[400px]">
        <RefreshCw className="animate-spin text-blue-500" size={24} />
        <span className="ml-2 text-slate-400">Loading admin panel...</span>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Shield className="text-brand-600 dark:text-brand-500" size={32} />
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Administration</h1>
      </div>

      {/* Database Statistics */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-800 flex items-center gap-2">
          <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg text-brand-600 dark:text-brand-500">
            <Database size={18} />
          </div>
          <h2 className="font-semibold text-gray-900 dark:text-white">Database Statistics</h2>
        </div>
        
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
              <div className="text-2xl font-bold text-brand-600 dark:text-brand-400">{stats?.games || 0}</div>
              <div className="text-sm text-gray-500 dark:text-gray-400">Total Games</div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">{stats?.elo_ratings || 0}</div>
              <div className="text-sm text-gray-500 dark:text-gray-400">ELO Ratings</div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">{stats?.elo_history || 0}</div>
              <div className="text-sm text-gray-500 dark:text-gray-400">ELO History Records</div>
            </div>
          </div>
        </div>
      </div>

      {/* Danger Zone */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-red-200 dark:border-red-800 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-red-100 dark:border-red-800 flex items-center gap-2">
          <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded-lg text-red-600 dark:text-red-400">
            <AlertTriangle size={18} />
          </div>
          <h2 className="font-semibold text-red-600 dark:text-red-400">Danger Zone</h2>
        </div>
        
        <div className="p-6">
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg mb-4 border border-red-200 dark:border-red-800">
            <p className="text-red-700 dark:text-red-300 text-sm mb-2">
              ⚠️ <strong>Warning:</strong> This action cannot be undone!
            </p>
            <p className="text-gray-600 dark:text-gray-300 text-sm">
              This will permanently delete all games, ELO ratings, and match history from the database.
            </p>
          </div>

          <button
            onClick={() => setShowModal(true)}
            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
          >
            <Trash2 size={18} />
            Reset Database
          </button>
        </div>
      </div>

      {/* Confirmation Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-xl p-6 max-w-md w-full mx-4 shadow-xl">
            <div className="flex items-center gap-2 mb-4">
              <AlertTriangle className="text-red-500" size={24} />
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">Confirm Database Reset</h3>
            </div>

            <div className="mb-4">
              <p className="text-gray-600 dark:text-gray-300 mb-2">
                Are you absolutely sure? This will permanently delete:
              </p>
              <ul className="text-sm text-gray-500 dark:text-gray-400 space-y-1">
                <li>• {stats?.games || 0} games</li>
                <li>• {stats?.elo_ratings || 0} ELO ratings</li>
                <li>• {stats?.elo_history || 0} history records</li>
              </ul>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Type "DELETE" to confirm:
              </label>
              <input
                type="text"
                value={confirmText}
                onChange={(e) => setConfirmText(e.target.value)}
                className="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-2 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-red-500"
                placeholder="Type DELETE here"
              />
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowModal(false);
                  setConfirmText('');
                }}
                className="flex-1 bg-gray-500 hover:bg-gray-600 text-white py-2 rounded-lg transition-colors"
                disabled={isResetting}
              >
                Cancel
              </button>
              <button
                onClick={handleResetDatabase}
                disabled={isResetting || confirmText !== 'DELETE'}
                className="flex-1 bg-red-600 hover:bg-red-700 disabled:bg-red-800 disabled:cursor-not-allowed text-white py-2 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {isResetting ? (
                  <>
                    <RefreshCw className="animate-spin" size={16} />
                    Resetting...
                  </>
                ) : (
                  <>
                    <Trash2 size={16} />
                    Reset Database
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Admin;