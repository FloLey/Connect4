import React, { useState, useEffect, useRef } from 'react'; // <--- Added useRef
import { 
  Trophy, Play, Square, Settings, 
  Cpu, Activity, CheckCircle2, AlertCircle, RefreshCw, Clock, ArrowRight as ArrowIcon
} from 'lucide-react';
import { 
  getModels, createTournament, createEvaluationTournament, startTournament, 
  stopTournament, getCurrentTournament, getActiveGames,
  pauseTournament, resumeTournament, updateTournamentConfig 
} from '../api/client';
import { Link } from 'react-router-dom';
import MiniGameBoard from '../components/MiniGameBoard'; // Import new component
import { useDatabase } from '../context/DatabaseContext';

const Tournament = () => {
  const { dbEnv } = useDatabase();
  // Config State
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [rounds, setRounds] = useState(1);
  const [concurrency, setConcurrency] = useState(2);
  const [mode, setMode] = useState('ROUND_ROBIN'); // 'ROUND_ROBIN' or 'EVALUATION'
  const [targetModel, setTargetModel] = useState('');
  
  // Live State
  const [activeTournament, setActiveTournament] = useState(null);
  const [activeGames, setActiveGames] = useState([]);
  const [loading, setLoading] = useState(true);
  
  // Pause/Resume State
  const [editingConcurrency, setEditingConcurrency] = useState(2);
  const [isUpdatingConfig, setIsUpdatingConfig] = useState(false);

  // 2. Add Refs
  // Stores the ID of a tournament the user has explicitly closed/reset
  const dismissedIdRef = useRef(null);
  // NEW: Track what tournament ID we are actively watching
  const viewingIdRef = useRef(null); 

  // --- Initialization & Polling ---
  useEffect(() => {
    // Reset local state on DB switch
    setActiveTournament(null);
    setActiveGames([]);
    setLoading(true);
    viewingIdRef.current = null;
    dismissedIdRef.current = null;

    fetchInitialData();
    const interval = setInterval(refreshStatus, 3000); // Poll every 3s
    return () => clearInterval(interval);
  }, [dbEnv]);

  const fetchInitialData = async () => {
    try {
      const [modelsData, tournamentData] = await Promise.all([
        getModels(),
        getCurrentTournament()
      ]);
      setAvailableModels(modelsData);

      // 3. Logic Update: Don't load if dismissed (edge case on re-mount)
      if (tournamentData && dismissedIdRef.current !== tournamentData.id) {
        // SMART LOGIC: Only set as active if we should be watching it
        // If tournament is COMPLETED/STOPPED on initial load, ignore it (show Setup)
        if (tournamentData.status === 'COMPLETED' || tournamentData.status === 'STOPPED') {
          // Don't set active tournament - user will see Setup screen
          viewingIdRef.current = null;
        } else {
          // Tournament is active (SETUP, IN_PROGRESS, PAUSED) - show it
          setActiveTournament(tournamentData);
          viewingIdRef.current = tournamentData.id;
          // Initialize editing concurrency with current config value
          setEditingConcurrency(tournamentData.config?.concurrency || 2);
        }
      }
      
      // Pre-select first 3 models if setup
      if (!tournamentData && modelsData.length >= 3) {
        setSelectedModels(modelsData.slice(0, 3).map(m => m.id));
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const refreshStatus = async () => {
    try {
      const t = await getCurrentTournament();
      if (!t) {
        // No tournament in DB - clear state
        setActiveTournament(null);
        viewingIdRef.current = null;
        return;
      }
      
      // 4. Logic Update: THE FIX
      // If the backend returns a tournament that matches our dismissed ID, ignore it.
      if (dismissedIdRef.current === t.id) {
        return; 
      }

      // SMART LOGIC: Only show "Finished" screen if we were actively watching this tournament
      if ((t.status === 'COMPLETED' || t.status === 'STOPPED') && viewingIdRef.current !== t.id) {
        // Tournament finished but we weren't watching it - ignore and show Setup
        return;
      }

      // If we get here, we should show this tournament
      setActiveTournament(t);
      
      // Track that we're now watching this tournament ID
      if (viewingIdRef.current !== t.id) {
        viewingIdRef.current = t.id;
      }
      
      // Update editing concurrency with current config value
      if (t) {
        setEditingConcurrency(t.config?.concurrency || 2);
      }
      
      if (t && t.status === 'IN_PROGRESS') {
        const games = await getActiveGames();
        
        // --- NEW LOGIC: STABLE SORT ---
        // Sort games by ID so they don't change positions in the grid during updates
        games.sort((a, b) => a.id - b.id);
        
        setActiveGames(games);
      }
    } catch (e) {
      console.error("Polling error", e);
    }
  };

  // --- Handlers ---
  // Reset target model when switching modes
  useEffect(() => {
    if (mode === 'ROUND_ROBIN') {
      setTargetModel('');
    }
  }, [mode]);

  const toggleModel = (id) => {
    setSelectedModels(prev => 
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
    
    // In evaluation mode, if we're selecting the target model, ensure it's in selected models
    if (mode === 'EVALUATION' && targetModel === id && !selectedModels.includes(id)) {
      setSelectedModels(prev => [...prev, id]);
    }
  };

  const handleCreate = async () => {
    if (selectedModels.length < 2) return alert("Select at least 2 models");
    if (mode === 'EVALUATION' && !targetModel) return alert("Select a target model for evaluation");
    
    setLoading(true);
    try {
      // 5. Logic Update: Reset dismissal ref for the NEW tournament
      dismissedIdRef.current = null;
      viewingIdRef.current = null;
      
      let t;
      if (mode === 'ROUND_ROBIN') {
        t = await createTournament(selectedModels, rounds, concurrency);
      } else {
        // For evaluation mode, filter out the target model from benchmarks
        const benchmarks = selectedModels.filter(m => m !== targetModel);
        t = await createEvaluationTournament(targetModel, benchmarks, rounds, concurrency);
      }
      
      setActiveTournament(t);
      viewingIdRef.current = t.id;
    } catch (e) {
      alert("Failed to create tournament");
    } finally {
      setLoading(false);
    }
  };

  const handleStart = async () => {
    if (!activeTournament) return;
    try {
      await startTournament(activeTournament.id);
      refreshStatus();
    } catch (e) {
      alert("Failed to start");
    }
  };

  const handleStop = async () => {
    if (!activeTournament) return;
    if (!confirm("Stop the tournament? Running games will finish, pending games will remain pending.")) return;
    try {
      await stopTournament(activeTournament.id);
      refreshStatus();
    } catch (e) {
      alert("Failed to stop");
    }
  };

  const handlePause = async () => {
    if (!activeTournament) return;
    try {
      await pauseTournament(activeTournament.id);
      // Optimistically update local state for instant feedback
      setActiveTournament(prev => ({...prev, status: 'PAUSED'}));
      refreshStatus();
    } catch (e) {
      alert("Failed to pause tournament");
    }
  };

  const handleResume = async () => {
    if (!activeTournament) return;
    try {
      await resumeTournament(activeTournament.id);
      // Optimistically update local state for instant feedback
      setActiveTournament(prev => ({...prev, status: 'IN_PROGRESS'}));
      refreshStatus();
    } catch (e) {
      alert("Failed to resume tournament");
    }
  };

  const handleUpdateConfig = async () => {
    if (!activeTournament) return;
    setIsUpdatingConfig(true);
    try {
      await updateTournamentConfig(activeTournament.id, editingConcurrency);
      // Update local tournament config
      setActiveTournament(prev => ({
        ...prev,
        config: {...prev.config, concurrency: editingConcurrency}
      }));
      alert("Configuration updated successfully");
    } catch (e) {
      alert("Failed to update configuration");
    } finally {
      setIsUpdatingConfig(false);
    }
  };

  const handleReset = () => {
    // 6. Logic Update: Mark current ID as dismissed
    if (activeTournament) {
      dismissedIdRef.current = activeTournament.id;
      viewingIdRef.current = null;
    }
    setActiveTournament(null); 
  };

  // --- Calculations ---
  const n = selectedModels.length;
  const matchesPerRound = mode === 'ROUND_ROBIN' ? n * (n - 1) : n * 2;
  const totalCalculated = matchesPerRound * rounds;

  // Helper to check if a game is snoozed due to rate limit
  const isSnoozed = (game) => {
    return game.status === 'PAUSED' && game.retry_after && new Date(game.retry_after) > new Date();
  };

  if (loading && !activeTournament) return <div className="p-10 text-center">Loading...</div>;

  // --- VIEW: ACTIVE TOURNAMENT ---
  if (activeTournament && activeTournament.status !== 'COMPLETED' && activeTournament.status !== 'STOPPED') {
    const pct = Math.round((activeTournament.completed / activeTournament.total) * 100) || 0;
    
    // SAFE ACCESS: Use ?.config?. to prevent crashes if config isn't loaded yet
    const config = activeTournament.config || {}; 

    return (
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold flex items-center gap-2 dark:text-white">
            <Trophy className="text-yellow-500" /> Tournament #{activeTournament.id}
          </h1>
          <div className="flex gap-2">
            {activeTournament.status === 'SETUP' && (
              <button onClick={handleStart} className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center gap-2">
                <Play size={18} /> Start Now
              </button>
            )}
            {activeTournament.status === 'IN_PROGRESS' && (
              <>
                <button onClick={handlePause} className="bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-lg flex items-center gap-2">
                  <Square size={18} /> Pause
                </button>
                <button onClick={handleStop} className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg flex items-center gap-2">
                  <Square size={18} /> Stop
                </button>
              </>
            )}
            {activeTournament.status === 'PAUSED' && (
              <button onClick={handleResume} className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center gap-2">
                <Play size={18} /> Resume
              </button>
            )}
          </div>
        </div>

        {/* Status Card */}
        <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-6 shadow-sm">
           <div className="flex justify-between text-sm mb-2 text-gray-500">
             <span>Progress</span>
             <span>{activeTournament.completed} / {activeTournament.total} Games</span>
           </div>
           <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-4 mb-6 overflow-hidden">
             <div 
               className="bg-brand-600 h-4 rounded-full transition-all duration-500 relative" 
               style={{ width: `${pct}%` }}
             >
                <div className="absolute inset-0 bg-white/20 animate-[shimmer_2s_infinite]"></div>
             </div>
           </div>
           
           <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
             <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-gray-500">Status</div>
                <div className="font-bold dark:text-white">{activeTournament.status}</div>
             </div>
             <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-gray-500">Concurrency</div>
                {activeTournament.status === 'PAUSED' ? (
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="font-bold dark:text-white">{editingConcurrency} Workers</span>
                      <button 
                        onClick={handleUpdateConfig}
                        disabled={isUpdatingConfig}
                        className="px-2 py-1 bg-brand-600 text-white text-xs rounded hover:bg-brand-700 disabled:opacity-50"
                      >
                        {isUpdatingConfig ? 'Saving...' : 'Save'}
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
                {/* FIXED: Added Optional Chaining here */}
                <div className="font-bold dark:text-white">{config.rounds || 1}</div>
             </div>
             <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-gray-500">Models</div>
                {/* FIXED: Added Optional Chaining here */}
                <div className="font-bold dark:text-white">{config.model_ids?.length || 0}</div>
             </div>
           </div>
        </div>

        {/* Active Games Grid */}
        <div className="space-y-4">
           <h2 className="text-lg font-semibold flex items-center gap-2 dark:text-white">
             <Activity size={20} className="text-brand-500" /> Live Matches
           </h2>
           
           {activeGames.length === 0 ? (
             <div className="p-8 text-center text-gray-500 bg-white dark:bg-gray-900 rounded-xl border border-dashed border-gray-300 dark:border-gray-700">
                {activeTournament.status === 'SETUP' ? "Waiting to start..." : "Spinning up workers..."}
             </div>
           ) : (
             <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
               {activeGames.map(game => (
                 <Link 
                   key={game.id} 
                   to={`/game/${game.id}`} 
                   className="block bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-4 hover:border-brand-500 dark:hover:border-brand-500 hover:shadow-md transition-all group"
                 >
                    {/* Header: ID + Move Count */}
                    <div className="flex justify-between items-center mb-3">
                       <span className="text-xs font-mono text-gray-400">#{game.id}</span>
                       <div className="flex gap-2">
                         {isSnoozed(game) && (
                           <span className="flex items-center gap-1 text-[10px] bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full animate-pulse">
                             <Clock size={10} /> Rate Limit Cooldown
                           </span>
                         )}
                         <span className="flex items-center gap-1 text-[10px] font-bold text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded-full">
                           <RefreshCw size={10} className={activeTournament.status === 'IN_PROGRESS' ? 'animate-spin' : ''}/>
                           {game.move_count || 0}
                         </span>
                       </div>
                    </div>

                    {/* Body: Board + Players */}
                    <div className="flex gap-4">
                      {/* Mini Board */}
                      <div className="shrink-0">
                         <MiniGameBoard board={game.board} />
                      </div>

                      {/* Players */}
                      <div className="flex-1 flex flex-col justify-center gap-2 overflow-hidden">
                        <div className="flex items-center gap-2 min-w-0">
                          <div className="w-2 h-2 rounded-full bg-red-500 shrink-0 shadow-sm" />
                          <span className="text-xs font-medium text-gray-700 dark:text-gray-300 truncate" title={game.player_1}>
                            {game.player_1}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 min-w-0">
                          <div className="w-2 h-2 rounded-full bg-yellow-400 shrink-0 shadow-sm" />
                          <span className="text-xs font-medium text-gray-700 dark:text-gray-300 truncate" title={game.player_2}>
                            {game.player_2}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Footer: Status Indicator */}
                    <div className="mt-3 flex items-center justify-between border-t border-gray-50 dark:border-gray-800 pt-2">
                        {isSnoozed(game) ? (
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
               ))}
             </div>
           )}
        </div>
      </div>
    );
  }

  // --- VIEW: COMPLETED / STOPPED ---
  if (activeTournament) {
    return (
      <div className="max-w-2xl mx-auto text-center space-y-6 pt-10">
        <div className="bg-white dark:bg-gray-900 p-8 rounded-2xl border border-gray-200 dark:border-gray-800 shadow-sm">
           <CheckCircle2 size={64} className="mx-auto text-green-500 mb-4" />
           <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Tournament Finished</h1>
           <p className="text-gray-500 mb-6">All {activeTournament.total} scheduled matches have been processed.</p>
           
           <div className="flex justify-center gap-4">
             <button onClick={handleReset} className="px-6 py-2 bg-brand-600 text-white rounded-lg hover:bg-brand-700">
               Start New Tournament
             </button>
             <Link to="/statistics" className="px-6 py-2 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white rounded-lg hover:bg-gray-200">
               View Leaderboard
             </Link>
           </div>
        </div>
      </div>
    );
  }

  // --- VIEW: SETUP FORM ---
  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold dark:text-white flex items-center gap-2">
           <Settings className="text-brand-600" /> Tournament Setup
        </h1>
        <p className="text-gray-500 dark:text-gray-400">
          {mode === 'ROUND_ROBIN' 
            ? 'Configure an automated Round Robin tournament.' 
            : 'Configure a focused Model Evaluation tournament.'}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
         {/* Left: Configuration */}
         <div className="lg:col-span-2 space-y-6">
            
            {/* Mode Toggle */}
            <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm">
               <h3 className="font-semibold mb-4 flex items-center gap-2 dark:text-white">
                 <Settings size={18} /> Tournament Mode
               </h3>
               <div className="flex gap-2">
                 <button
                   onClick={() => setMode('ROUND_ROBIN')}
                   className={`flex-1 py-3 rounded-lg border transition-all ${mode === 'ROUND_ROBIN' 
                     ? 'bg-brand-50 border-brand-500 dark:bg-brand-900/20 text-brand-700 dark:text-brand-300' 
                     : 'bg-gray-50 border-gray-200 dark:bg-gray-800 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-gray-300'}`}
                 >
                   <div className="font-medium">Full Arena</div>
                   <div className="text-xs mt-1">Round Robin (All vs All)</div>
                 </button>
                 <button
                   onClick={() => setMode('EVALUATION')}
                   className={`flex-1 py-3 rounded-lg border transition-all ${mode === 'EVALUATION' 
                     ? 'bg-brand-50 border-brand-500 dark:bg-brand-900/20 text-brand-700 dark:text-brand-300' 
                     : 'bg-gray-50 border-gray-200 dark:bg-gray-800 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-gray-300'}`}
                 >
                   <div className="font-medium">Model Evaluation</div>
                   <div className="text-xs mt-1">Pivot Mode (Target vs Benchmarks)</div>
                 </button>
               </div>
            </div>
            
            {/* Model Selection */}
            <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm">
               <h3 className="font-semibold mb-4 flex items-center gap-2 dark:text-white">
                 <Cpu size={18} /> Select Models ({selectedModels.length})
               </h3>
               <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-60 overflow-y-auto pr-2">
                 {availableModels.map(m => (
                   <div 
                     key={m.id}
                     onClick={() => toggleModel(m.id)}
                     className={`p-3 rounded-lg border cursor-pointer transition-all flex items-center justify-between ${
                       selectedModels.includes(m.id)
                         ? 'bg-brand-50 border-brand-500 dark:bg-brand-900/20'
                         : 'bg-gray-50 border-gray-200 dark:bg-gray-800 dark:border-gray-700 hover:border-gray-300'
                     }`}
                   >
                     <div className="text-sm font-medium dark:text-gray-200">{m.label}</div>
                     {selectedModels.includes(m.id) && <CheckCircle2 size={16} className="text-brand-600" />}
                   </div>
                 ))}
               </div>
               
               {/* Target Model Selector for Evaluation Mode */}
               {mode === 'EVALUATION' && (
                 <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-800">
                   <h4 className="font-semibold mb-3 dark:text-white">Target Model (Being Evaluated)</h4>
                   <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                     {availableModels.map(m => (
                       <div 
                         key={`target-${m.id}`}
                         onClick={() => setTargetModel(m.id)}
                         className={`p-3 rounded-lg border cursor-pointer transition-all flex items-center justify-between ${
                           targetModel === m.id
                             ? 'bg-brand-50 border-brand-500 dark:bg-brand-900/20'
                             : 'bg-gray-50 border-gray-200 dark:bg-gray-800 dark:border-gray-700 hover:border-gray-300'
                         }`}
                       >
                         <div className="text-sm font-medium dark:text-gray-200">{m.label}</div>
                         {targetModel === m.id && <CheckCircle2 size={16} className="text-brand-600" />}
                       </div>
                     ))}
                   </div>
                   <p className="text-xs text-gray-500 mt-2">
                     The target model will play 2 games (as P1 and P2) against each benchmark model.
                   </p>
                 </div>
               )}
            </div>

            {/* Sliders */}
            <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm space-y-6">
               <div>
                  <div className="flex justify-between mb-2">
                    <label className="font-semibold dark:text-white flex items-center gap-2"><RefreshCw size={18}/> Rounds (Cycles)</label>
                    <span className="text-brand-600 font-mono font-bold">{rounds}</span>
                  </div>
                  <input 
                    type="range" min="1" max="10" step="1"
                    value={rounds} onChange={(e) => setRounds(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
                  />
                  <p className="text-xs text-gray-500 mt-1">Number of times every pair plays each other (swapping sides).</p>
               </div>

               <div>
                  <div className="flex justify-between mb-2">
                    <label className="font-semibold dark:text-white flex items-center gap-2"><Activity size={18}/> Concurrency</label>
                    <span className="text-brand-600 font-mono font-bold">{concurrency}</span>
                  </div>
                  <input 
                    type="range" min="1" max="100" step="1"
                    value={concurrency} onChange={(e) => setConcurrency(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
                  />
                  <p className="text-xs text-gray-500 mt-1">Max simultaneous games. Higher = Faster but heavier on API/DB.</p>
               </div>
            </div>
         </div>

         {/* Right: Summary */}
         <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm sticky top-24">
               <h3 className="font-bold text-lg mb-4 dark:text-white">Summary</h3>
               
               <div className="space-y-3 text-sm border-b border-gray-100 dark:border-gray-800 pb-4 mb-4">
                 <div className="flex justify-between">
                   <span className="text-gray-500">Models</span>
                   <span className="font-mono dark:text-gray-200">{n}</span>
                 </div>
                 <div className="flex justify-between">
                   <span className="text-gray-500">Matchups ({mode === 'ROUND_ROBIN' ? 'Round Robin' : 'Evaluation'})</span>
                   <span className="font-mono dark:text-gray-200">{matchesPerRound}</span>
                 </div>
                 <div className="flex justify-between">
                   <span className="text-gray-500">Total Rounds</span>
                   <span className="font-mono dark:text-gray-200">x {rounds}</span>
                 </div>
               </div>

               <div className="flex justify-between items-center mb-6">
                 <span className="font-bold text-gray-900 dark:text-white">Total Games</span>
                 <span className="text-xl font-bold text-brand-600">{totalCalculated}</span>
               </div>

               {n < 2 || (mode === 'EVALUATION' && !targetModel) ? (
                 <div className="text-sm text-red-500 flex items-center gap-2 bg-red-50 p-3 rounded-lg">
                   <AlertCircle size={16} /> 
                   {n < 2 ? 'Select at least 2 models' : 'Select a target model for evaluation'}
                 </div>
               ) : (
                 <button 
                   onClick={handleCreate}
                   className="w-full py-3 bg-brand-600 hover:bg-brand-700 text-white font-bold rounded-xl shadow-lg hover:shadow-xl transition-all flex items-center justify-center gap-2"
                 >
                   {loading ? "Creating..." : <>Create & Queue <ArrowIcon size={18}/></>}
                 </button>
               )}
            </div>
         </div>
      </div>
    </div>
  );
};

export default Tournament;