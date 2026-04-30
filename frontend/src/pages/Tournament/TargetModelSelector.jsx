import { CheckCircle2 } from 'lucide-react';

const TargetModelSelector = ({ availableModels, targetModel, onChange }) => (
  <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-800">
    <h4 className="font-semibold mb-3 dark:text-white">Target Model (Being Evaluated)</h4>
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {availableModels.map((m) => (
        <div
          key={`target-${m.id}`}
          onClick={() => onChange(m.id)}
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
);

export default TargetModelSelector;
