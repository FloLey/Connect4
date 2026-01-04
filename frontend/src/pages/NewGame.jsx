import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { createGame, getModels } from '../api/client';
import { Bot, User, Sword, Shuffle } from 'lucide-react';
import { useDatabase } from '../context/DatabaseContext';
import { saveGameTokenWithTimestamp } from '../utils/tokenCleanup';

const NewGame = () => {
  const { dbEnv } = useDatabase();
  const navigate = useNavigate();
  const [models, setModels] = useState([]); // Dynamic List
  const [loadingModels, setLoadingModels] = useState(true);
  
  // Form State
  const [p1Type, setP1Type] = useState('human');
  const [p1Model, setP1Model] = useState('');
  const [p2Type, setP2Type] = useState('ai');
  const [p2Model, setP2Model] = useState('');
  
  const [loading, setLoading] = useState(false);

  // 1. Fetch Models on Mount and when dbEnv changes
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await getModels();
        setModels(data);
        if (data.length > 0) {
          setP1Model(data[0].id);
          setP2Model(data[0].id);
        }
      } catch (error) {
        console.error("Failed to load models", error);
      } finally {
        setLoadingModels(false);
      }
    };
    fetchModels();
  }, [dbEnv]);

  const handleRandomize = () => {
    if (models.length < 2) return;

    // 1. Pick first random model
    const randomIndex1 = Math.floor(Math.random() * models.length);
    const model1 = models[randomIndex1];

    // 2. Pick second random model (ensure it's different)
    let model2;
    do {
      const randomIndex2 = Math.floor(Math.random() * models.length);
      model2 = models[randomIndex2];
    } while (model2.id === model1.id);

    // 3. Update all states
    setP1Type('ai');
    setP2Type('ai');
    setP1Model(model1.id);
    setP2Model(model2.id);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const player1 = p1Type === 'human' ? 'human' : p1Model;
      const player2 = p2Type === 'human' ? 'human' : p2Model;
      
      const game = await createGame(player1, player2);
      
      // Save tokens if they exist in response (with timestamps for cleanup)
      if (game.player_1_token) saveGameTokenWithTimestamp(game.id, game.player_1_token);
      if (game.player_2_token) saveGameTokenWithTimestamp(game.id, game.player_2_token);
      
      navigate(`/game/${game.id}`);
    } catch (error) {
      console.error(error);
      alert('Failed to start game');
    } finally {
      setLoading(false);
    }
  };

  // 2. Dynamic Select Component
  const ModelSelect = ({ value, onChange }) => {
    if (loadingModels) return <div className="text-gray-500 dark:text-gray-400 text-sm">Loading models...</div>;
    
    // Group by Provider
    const groups = models.reduce((acc, model) => {
      const p = model.provider.charAt(0).toUpperCase() + model.provider.slice(1);
      if (!acc[p]) acc[p] = [];
      acc[p].push(model);
      return acc;
    }, {});

    return (
      <select 
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-4 py-3 text-gray-900 dark:text-white focus:ring-2 focus:ring-brand-500 outline-none"
      >
        {Object.keys(groups).map(provider => (
          <optgroup key={provider} label={provider}>
            {groups[provider].map(m => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </optgroup>
        ))}
      </select>
    );
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm overflow-hidden">
        <div className="p-8 border-b border-gray-100 dark:border-gray-800 text-center">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Setup New Match</h2>
          <p className="text-gray-500 dark:text-gray-400">Configure your match settings</p>
        </div>
        
        <form onSubmit={handleSubmit} className="p-8 space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 relative">
            <div className="hidden md:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 bg-gray-100 dark:bg-gray-800 rounded-full items-center justify-center border-2 border-gray-200 dark:border-gray-700 z-10 font-bold text-gray-500 dark:text-gray-400">VS</div>

            {/* Player 1 Config */}
            <div className="space-y-4">
              <label className="block text-sm font-medium text-red-500 dark:text-red-400 uppercase tracking-wider">Player 1 (Red)</label>
              <div className="flex gap-2 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg">
                <button type="button" onClick={() => setP1Type('human')} className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm transition ${p1Type === 'human' ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm' : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'}`}><User size={16} /> Human</button>
                <button type="button" onClick={() => setP1Type('ai')} className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm transition ${p1Type === 'ai' ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm' : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'}`}><Bot size={16} /> AI</button>
              </div>
              {p1Type === 'ai' && <ModelSelect value={p1Model} onChange={setP1Model} />}
            </div>

            {/* Player 2 Config */}
            <div className="space-y-4">
              <label className="block text-sm font-medium text-yellow-500 dark:text-yellow-400 uppercase tracking-wider">Player 2 (Yellow)</label>
              <div className="flex gap-2 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg">
                <button type="button" onClick={() => setP2Type('human')} className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm transition ${p2Type === 'human' ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm' : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'}`}><User size={16} /> Human</button>
                <button type="button" onClick={() => setP2Type('ai')} className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-md text-sm transition ${p2Type === 'ai' ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm' : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'}`}><Bot size={16} /> AI</button>
              </div>
              {p2Type === 'ai' && <ModelSelect value={p2Model} onChange={setP2Model} />}
            </div>
          </div>

          {/* Randomize Button */}
          <button 
            type="button"
            onClick={handleRandomize}
            disabled={loading || loadingModels}
            className="w-full mb-4 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200 font-medium py-3 rounded-lg border border-gray-200 dark:border-gray-700 flex items-center justify-center gap-2 transition-colors"
          >
            <Shuffle size={20} />
            Randomize Matchup
          </button>

          <button type="submit" disabled={loading || loadingModels} className="w-full bg-brand-600 hover:bg-brand-700 disabled:bg-gray-400 text-white font-medium py-3 rounded-lg shadow-sm flex items-center justify-center gap-2 transition-colors">
            {loading ? <span className="animate-pulse">Starting Match...</span> : <><Sword size={20} /> Start Game</>}
          </button>
        </form>
      </div>
    </div>
  );
};

export default NewGame;