import { useEffect, useState } from 'react';
import {
  Activity,
  AlertCircle,
  ArrowRight as ArrowIcon,
  CheckCircle2,
  Cpu,
  RefreshCw,
  Settings,
} from 'lucide-react';

import TargetModelSelector from './TargetModelSelector';

const TournamentSetupForm = ({ availableModels, onCreate }) => {
  const [mode, setMode] = useState('ROUND_ROBIN');
  const [selectedModels, setSelectedModels] = useState([]);
  const [targetModel, setTargetModel] = useState('');
  const [rounds, setRounds] = useState(1);
  const [concurrency, setConcurrency] = useState(2);

  // Pre-select first 3 models when the registry first arrives.
  useEffect(() => {
    if (availableModels.length >= 3 && selectedModels.length === 0) {
      setSelectedModels(availableModels.slice(0, 3).map((m) => m.id));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableModels]);

  // Reset target model when leaving evaluation mode.
  useEffect(() => {
    if (mode === 'ROUND_ROBIN') {
      setTargetModel('');
    }
  }, [mode]);

  const toggleModel = (id) => {
    setSelectedModels((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const n = selectedModels.length;
  const matchesPerRound = mode === 'ROUND_ROBIN' ? n * (n - 1) : n * 2;
  const totalCalculated = matchesPerRound * rounds;
  const canCreate = n >= 2 && (mode === 'ROUND_ROBIN' || !!targetModel);

  const handleCreate = () => {
    onCreate({ mode, models: selectedModels, rounds, concurrency, targetModel });
  };

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
        <div className="lg:col-span-2 space-y-6">
          {/* Mode toggle */}
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

          {/* Model selection */}
          <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm">
            <h3 className="font-semibold mb-4 flex items-center gap-2 dark:text-white">
              <Cpu size={18} /> Select Models ({selectedModels.length})
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-60 overflow-y-auto pr-2">
              {availableModels.map((m) => (
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
                  {selectedModels.includes(m.id) && (
                    <CheckCircle2 size={16} className="text-brand-600" />
                  )}
                </div>
              ))}
            </div>

            {mode === 'EVALUATION' && (
              <TargetModelSelector
                availableModels={availableModels}
                targetModel={targetModel}
                onChange={setTargetModel}
              />
            )}
          </div>

          {/* Sliders */}
          <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm space-y-6">
            <div>
              <div className="flex justify-between mb-2">
                <label className="font-semibold dark:text-white flex items-center gap-2">
                  <RefreshCw size={18} /> Rounds (Cycles)
                </label>
                <span className="text-brand-600 font-mono font-bold">{rounds}</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={rounds}
                onChange={(e) => setRounds(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
              />
              <p className="text-xs text-gray-500 mt-1">Number of times every pair plays each other (swapping sides).</p>
            </div>

            <div>
              <div className="flex justify-between mb-2">
                <label className="font-semibold dark:text-white flex items-center gap-2">
                  <Activity size={18} /> Concurrency
                </label>
                <span className="text-brand-600 font-mono font-bold">{concurrency}</span>
              </div>
              <input
                type="range"
                min="1"
                max="100"
                step="1"
                value={concurrency}
                onChange={(e) => setConcurrency(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
              />
              <p className="text-xs text-gray-500 mt-1">Max simultaneous games. Higher = Faster but heavier on API/DB.</p>
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-900 p-6 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm sticky top-24">
            <h3 className="font-bold text-lg mb-4 dark:text-white">Summary</h3>
            <div className="space-y-3 text-sm border-b border-gray-100 dark:border-gray-800 pb-4 mb-4">
              <div className="flex justify-between">
                <span className="text-gray-500">Models</span>
                <span className="font-mono dark:text-gray-200">{n}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">
                  Matchups ({mode === 'ROUND_ROBIN' ? 'Round Robin' : 'Evaluation'})
                </span>
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
            {!canCreate ? (
              <div className="text-sm text-red-500 flex items-center gap-2 bg-red-50 p-3 rounded-lg">
                <AlertCircle size={16} />
                {n < 2 ? 'Select at least 2 models' : 'Select a target model for evaluation'}
              </div>
            ) : (
              <button
                onClick={handleCreate}
                className="w-full py-3 bg-brand-600 hover:bg-brand-700 text-white font-bold rounded-xl shadow-lg hover:shadow-xl transition-all flex items-center justify-center gap-2"
              >
                Create &amp; Queue <ArrowIcon size={18} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TournamentSetupForm;
