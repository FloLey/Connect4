import React, { useEffect, useState } from 'react';
import { getLeaderboard, getMatrix, getHistoryPlot } from '../api/client';
import { 
  BarChart2, RefreshCw, Zap, Clock, Coins, Activity, 
  Maximize2, X, ChevronDown, ChevronUp, Table as TableIcon, Filter
} from 'lucide-react';
import { useDatabase } from '../context/DatabaseContext';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, LabelList, Brush, ReferenceLine
} from 'recharts';

// --- Utility: Deterministic Colors for Models ---
const COLORS = [
  '#2563eb', // blue-600
  '#dc2626', // red-600
  '#ca8a04', // yellow-600
  '#16a34a', // green-600
  '#9333ea', // purple-600
  '#db2777', // pink-600
  '#0891b2', // cyan-600
  '#ea580c', // orange-600
  '#4b5563', // gray-600 (human)
  '#0d9488', // teal-600
  '#be185d', // pink-700
  '#4338ca', // indigo-700
  '#15803d', // green-700
  '#b45309', // amber-700
];

const getModelColor = (modelName, index) => {
  if (modelName === 'human') return '#4b5563';
  let hash = 0;
  for (let i = 0; i < modelName.length; i++) {
    hash = modelName.charCodeAt(i) + ((hash << 5) - hash);
  }
  const positiveHash = Math.abs(hash);
  return COLORS[positiveHash % COLORS.length];
};

// --- Custom Tooltip for Scatter Plots ---
const ScatterTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-white dark:bg-gray-900 p-3 border border-gray-200 dark:border-gray-700 shadow-lg rounded-lg text-xs z-50">
        <p className="font-bold text-gray-900 dark:text-white mb-2 text-sm">{data.model_name}</p>
        <div className="space-y-1">
          <p className="text-gray-500">ELO: <span className="font-mono text-gray-900 dark:text-white">{Math.round(data.rating)}</span></p>
          {data.avg_cost_per_game !== undefined && (
            <p className="text-emerald-600">Cost: <span className="font-mono">${data.avg_cost_per_game}/game</span></p>
          )}
          {data.mean_time_per_move !== undefined && (
            <p className="text-blue-600">Time: <span className="font-mono">{data.mean_time_per_move}s/move</span></p>
          )}
          {data.mean_tokens_out_per_move !== undefined && (
            <p className="text-purple-600">Output: <span className="font-mono">{data.mean_tokens_out_per_move} toks/move</span></p>
          )}
        </div>
      </div>
    );
  }
  return null;
};

// --- Chart Sub-Components ---

const MatrixView = ({ matrix }) => {
  if (!matrix) return null;
  return (
    <div className="overflow-x-auto h-full">
       <table className="w-full text-xs border-collapse h-full">
          <thead>
            <tr>
              <th className="p-2 bg-gray-50 dark:bg-gray-800 text-left min-w-[100px] sticky top-0 z-10">Model</th>
              {matrix.models.map(m => (
                <th key={m} className="p-2 bg-gray-50 dark:bg-gray-800 font-mono rotate-0 whitespace-nowrap overflow-hidden text-ellipsis max-w-[80px] sticky top-0 z-10" title={m}>
                  {m}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.models.map(rowModel => (
              <tr key={rowModel}>
                <td className="p-2 font-medium bg-gray-50 dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 truncate max-w-[120px]" title={rowModel}>
                  {rowModel}
                </td>
                {matrix.models.map(colModel => {
                  const cell = matrix.grid[rowModel][colModel];
                  if (rowModel === colModel) return <td key={colModel} className="bg-gray-100 dark:bg-gray-800"></td>;
                  if (cell.total === 0) return <td key={colModel} className="text-center text-gray-300 dark:text-gray-700">-</td>;

                  let bgColor;
                  const rate = cell.win_rate;
                  if (rate > 55) bgColor = `rgba(34, 197, 94, ${(rate - 50) / 50 * 0.5 + 0.2})`; // Green
                  else if (rate < 45) bgColor = `rgba(239, 68, 68, ${(50 - rate) / 50 * 0.5 + 0.2})`; // Red
                  else bgColor = 'rgba(107, 114, 128, 0.1)'; // Neutral

                  return (
                    <td 
                      key={colModel} 
                      className="text-center p-2 border border-gray-100 dark:border-gray-800 cursor-help transition-opacity hover:opacity-80"
                      style={{ backgroundColor: bgColor }}
                      title={`${rowModel} vs ${colModel}\nWins: ${cell.wins}, Losses: ${cell.losses}, Draws: ${cell.draws}\nTotal: ${cell.total}`}
                    >
                       <span className="font-bold text-gray-800 dark:text-gray-200">{Math.round(rate)}%</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
    </div>
  );
};

const ScatterPlotView = ({ data, xKey, yKey, name, unit, fill, xLabel }) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
        <XAxis type="number" dataKey={xKey} name={name} unit={unit} label={{ value: xLabel, position: 'bottom', offset: 0 }} stroke="#9ca3af" />
        <YAxis type="number" dataKey={yKey} name="ELO" domain={['auto', 'auto']} stroke="#9ca3af" />
        <ReTooltip cursor={{ strokeDasharray: '3 3' }} content={<ScatterTooltip />} />
        <Scatter name="Models" data={data} fill={fill}>
           {data.map((entry, index) => (
             <cell key={`cell-${index}`} fill={getModelColor(entry.model_name, index)} />
           ))}
           <LabelList dataKey="model_name" position="top" style={{ fontSize: '10px', fill: '#6b7280' }} />
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
};

// --- UPDATED: Cleaner History Plot ---
const HistoryPlotView = ({ data, models, visibleModels, toggleModel }) => {
  return (
    <div className="flex flex-col h-full">
      {/* Legend / Filter Control */}
      <div className="flex flex-wrap gap-2 mb-4 px-2 max-h-20 overflow-y-auto">
        {models.map((modelName, index) => (
          <button
            key={modelName}
            onClick={() => toggleModel(modelName)}
            className={`px-2 py-1 rounded text-xs font-medium border transition-all flex items-center gap-1 ${
              visibleModels.includes(modelName)
                ? 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 shadow-sm opacity-100'
                : 'bg-transparent border-transparent opacity-40 hover:opacity-70'
            }`}
            style={{ color: getModelColor(modelName, index) }}
          >
            <div 
              className="w-2 h-2 rounded-full" 
              style={{ backgroundColor: visibleModels.includes(modelName) ? getModelColor(modelName, index) : 'gray' }} 
            />
            {modelName}
          </button>
        ))}
      </div>

      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} />
            
            {/* UPDATED X-AXIS */}
            <XAxis 
              dataKey="match_number" 
              type="number" 
              domain={[0, 'auto']} // Forces start at 0
              stroke="#9ca3af" 
              tick={{fontSize: 12}}
              allowDecimals={false}
              label={{ value: 'Games Played (Per Model)', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 12 }} 
            />
            
            <YAxis 
              domain={['auto', 'auto']} 
              stroke="#9ca3af" 
              tick={{fontSize: 12}}
              width={40}
              allowDecimals={false} 
            />
            
            {/* Baseline Reference */}
            <ReferenceLine y={1200} stroke="#9ca3af" strokeDasharray="3 3" />

            {/* Tooltip formatter update */}
            <ReTooltip 
              contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px', color: '#fff', fontSize: '12px' }}
              labelFormatter={(value) => `Game #${value}`}
            />
            
            {models.map((modelName, index) => (
              visibleModels.includes(modelName) && (
                <Line 
                  key={modelName}
                  type="linear" 
                  dataKey={modelName} 
                  stroke={getModelColor(modelName, index)} 
                  strokeWidth={2}
                  dot={false} 
                  activeDot={{ r: 6, strokeWidth: 0 }} 
                  
                  // CRITICAL: Connects dots across empty data points
                  connectNulls={true} 
                  
                  isAnimationActive={false} 
                />
              )
            ))}
            
            <Brush 
              dataKey="match_number" 
              height={30} 
              stroke="#4b5563" 
              fill="transparent" 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const Statistics = () => {
  const { dbEnv } = useDatabase();
  const [leaderboard, setLeaderboard] = useState([]);
  const [matrix, setMatrix] = useState(null);
  const [historyData, setHistoryData] = useState([]);
  const [loading, setLoading] = useState(true);
  
  // UI States
  const [showTable, setShowTable] = useState(false);
  const [expandedChart, setExpandedChart] = useState(null);
  
  // Filtering for History Chart
  const [allModels, setAllModels] = useState([]);
  const [visibleModels, setVisibleModels] = useState([]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [lbData, matrixData, plotData] = await Promise.all([
        getLeaderboard(),
        getMatrix(),
        getHistoryPlot() // <--- Use the new optimized endpoint
      ]);
      setLeaderboard(lbData);
      setMatrix(matrixData);
      setHistoryData(plotData); // <--- No more processHistoryData needed!
      
      const modelsList = lbData.map(m => m.model_name);
      setAllModels(modelsList);
      if (visibleModels.length === 0) setVisibleModels(modelsList);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };


  useEffect(() => {
    setLoading(true);
    setLeaderboard([]); // Clear stale data
    setMatrix(null);
    setHistoryData([]);
    
    fetchData();
  }, [dbEnv]);

  const toggleModelVisibility = (modelName) => {
    setVisibleModels(prev => 
      prev.includes(modelName) 
        ? prev.filter(m => m !== modelName)
        : [...prev, modelName]
    );
  };

  const aiOnlyLeaderboard = leaderboard.filter(m => m.model_name !== 'human');

  // --- Modal Renderer ---
  const renderExpandedContent = () => {
    switch (expandedChart) {
      case 'matrix': return <MatrixView matrix={matrix} />;
      case 'cost': return <ScatterPlotView data={aiOnlyLeaderboard} xKey="avg_cost_per_game" yKey="rating" name="Cost" unit="$" fill="#10b981" xLabel="$/Game" />;
      case 'time': return <ScatterPlotView data={aiOnlyLeaderboard} xKey="mean_time_per_move" yKey="rating" name="Time" unit="s" fill="#3b82f6" xLabel="Seconds/Move" />;
      case 'tokens': return <ScatterPlotView data={aiOnlyLeaderboard} xKey="mean_tokens_out_per_move" yKey="rating" name="Tokens" unit="T" fill="#8b5cf6" xLabel="Tokens/Move" />;
      case 'history': return <HistoryPlotView data={historyData} models={allModels} visibleModels={visibleModels} toggleModel={toggleModelVisibility} />;
      default: return null;
    }
  };

  // --- Helper to wrap charts in a card with expand button ---
  const ChartCard = ({ title, icon: Icon, color, id, children }) => (
    <div 
      className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm p-4 relative group flex flex-col h-[400px] transition-all hover:border-brand-300 dark:hover:border-brand-700"
      onClick={() => setExpandedChart(id)}
    >
      <div className={`flex items-center gap-2 mb-2 text-sm font-bold ${color} shrink-0`}>
        <Icon size={16} /> <span>{title}</span>
      </div>
      <div className="flex-1 min-h-0 cursor-pointer overflow-hidden">
        {children}
      </div>
      <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity z-10">
        <button className="p-1.5 bg-gray-100 dark:bg-gray-800 rounded-lg text-gray-500 hover:text-brand-600 shadow-sm border border-gray-200 dark:border-gray-700">
           <Maximize2 size={16} />
        </button>
      </div>
    </div>
  );

  return (
    <div className="max-w-[1400px] mx-auto space-y-6 pb-12">
      
      {/* --- Header & Controls --- */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <BarChart2 className="text-brand-600 dark:text-brand-500" size={32} />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Analytics Center</h1>
        </div>
        <button 
          onClick={fetchData} 
          className="p-2 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
        >
          <RefreshCw size={20} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {/* --- Collapsible FULL Leaderboard Table --- */}
      <div className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm overflow-hidden">
        <button 
          onClick={() => setShowTable(!showTable)}
          className="w-full px-6 py-4 flex items-center justify-between bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          <div className="flex items-center gap-2 text-gray-900 dark:text-white font-semibold">
            <TableIcon size={18} className="text-brand-600 dark:text-brand-400" />
            Full Model Leaderboard
          </div>
          {showTable ? <ChevronUp size={20} className="text-gray-500"/> : <ChevronDown size={20} className="text-gray-500"/>}
        </button>
        
        {showTable && (
          <div className="overflow-x-auto border-t border-gray-100 dark:border-gray-800">
            <table className="w-full text-xs text-left">
              <thead className="bg-gray-50 dark:bg-gray-800/50 text-gray-500 dark:text-gray-400 uppercase">
                <tr>
                  <th className="px-4 py-3 font-medium text-left">Model</th>
                  <th className="px-2 py-3 font-medium text-right">ELO</th>
                  <th className="px-2 py-3 font-medium text-right">Win%</th>
                  <th className="px-2 py-3 font-medium text-right">Games</th>
                  
                  {/* Stats */}
                  <th className="px-2 py-3 font-medium text-right text-gray-400">Time/Mv</th>
                  <th className="px-2 py-3 font-medium text-right text-gray-400">Moves/Game</th>
                  <th className="px-2 py-3 font-medium text-right text-gray-400">Tok/Mv</th>
                  <th className="px-2 py-3 font-medium text-right text-gray-400">Total Tok</th>
                  
                  {/* Economics */}
                  <th className="px-2 py-3 font-medium text-right text-emerald-600">$/Mv</th>
                  <th className="px-2 py-3 font-medium text-right text-emerald-600">$/Game</th>
                  <th className="px-2 py-3 font-medium text-right text-emerald-600">Total $</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                {leaderboard.map((row) => {
                  const total = row.wins + row.losses + row.draws;
                  const wr = total > 0 ? ((row.wins / total) * 100).toFixed(1) : 0;
                  return (
                    <tr key={row.model_name} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                      <td className="px-4 py-3 font-medium text-gray-900 dark:text-white truncate max-w-[150px]" title={row.model_name}>
                        {row.model_name}
                      </td>
                      <td className="px-2 py-3 text-right font-mono text-brand-600 dark:text-brand-400 font-bold">{Math.round(row.rating)}</td>
                      <td className="px-2 py-3 text-right">{wr}%</td>
                      <td className="px-2 py-3 text-right text-gray-500">{row.matches_played}</td>
                      
                      {/* Stats */}
                      <td className="px-2 py-3 text-right font-mono text-gray-600 dark:text-gray-400">{row.mean_time_per_move}s</td>
                      <td className="px-2 py-3 text-right font-mono text-gray-600 dark:text-gray-400">{row.avg_moves_per_game}</td>
                      <td className="px-2 py-3 text-right font-mono text-gray-600 dark:text-gray-400">{row.mean_tokens_out_per_move}</td>
                      <td className="px-2 py-3 text-right font-mono text-gray-600 dark:text-gray-400">{(row.total_tokens_out / 1000000).toFixed(2)}M</td>
                      
                      {/* Economics */}
                      <td className="px-2 py-3 text-right font-mono text-emerald-600 dark:text-emerald-500">${row.avg_cost_per_move}</td>
                      <td className="px-2 py-3 text-right font-mono text-emerald-600 dark:text-emerald-500">${row.avg_cost_per_game}</td>
                      <td className="px-2 py-3 text-right font-mono text-emerald-600 dark:text-emerald-500 font-bold">${row.total_cost}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* --- Matrix --- */}
      <ChartCard title="Win Rate Matrix" icon={Activity} color="text-brand-600" id="matrix">
        <MatrixView matrix={matrix} />
      </ChartCard>

      {/* --- Scatter Plots Row --- */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <ChartCard title="Price vs Performance" icon={Coins} color="text-emerald-600" id="cost">
          <ScatterPlotView data={aiOnlyLeaderboard} xKey="avg_cost_per_game" yKey="rating" name="Cost" unit="$" fill="#10b981" xLabel="$/Game" />
        </ChartCard>

        <ChartCard title="Speed vs Performance" icon={Clock} color="text-blue-600" id="time">
           <ScatterPlotView data={aiOnlyLeaderboard} xKey="mean_time_per_move" yKey="rating" name="Time" unit="s" fill="#3b82f6" xLabel="Seconds/Move" />
        </ChartCard>

        <ChartCard title="Verbosity vs Performance" icon={Zap} color="text-purple-600" id="tokens">
           <ScatterPlotView data={aiOnlyLeaderboard} xKey="mean_tokens_out_per_move" yKey="rating" name="Tokens" unit="T" fill="#8b5cf6" xLabel="Tokens/Move" />
        </ChartCard>
      </div>

      {/* --- History --- */}
      <div className="h-[500px] w-full">
        <ChartCard title="ELO Progression History" icon={Activity} color="text-gray-900 dark:text-white" id="history">
          <HistoryPlotView 
            data={historyData} 
            models={allModels} 
            visibleModels={visibleModels}
            toggleModel={toggleModelVisibility}
          />
        </ChartCard>
      </div>


      {/* --- EXPANDED MODAL --- */}
      {expandedChart && (
        <div className="fixed inset-0 z-[100] bg-white dark:bg-gray-950 p-6 flex flex-col">
          {/* Modal Header */}
          <div className="flex justify-between items-center mb-6 shrink-0">
             <div className="flex items-center gap-3">
               <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
                 <Maximize2 size={24} className="text-brand-600 dark:text-brand-400" />
               </div>
               <div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white capitalize">
                    {expandedChart === 'matrix' ? 'Win Rate Matrix' : expandedChart + ' Analysis'}
                  </h2>
                  <p className="text-gray-500">Detailed View</p>
               </div>
             </div>
             <button 
               onClick={() => setExpandedChart(null)}
               className="p-3 bg-gray-100 dark:bg-gray-800 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
             >
               <X size={24} className="text-gray-600 dark:text-gray-300" />
             </button>
          </div>

          {/* Modal Content */}
          <div className="flex-1 min-h-0 bg-gray-50 dark:bg-gray-900/50 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 overflow-hidden">
             {renderExpandedContent()}
          </div>
        </div>
      )}

    </div>
  );
};

export default Statistics;