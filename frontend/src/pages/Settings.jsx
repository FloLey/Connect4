import { useEffect, useState } from 'react';
import { Key, Sliders, RefreshCw, X, Save, Eye, EyeOff } from 'lucide-react';

import {
  getSettings,
  patchSettings,
  clearApiKey,
  clearTunable,
} from '../api/client';
import Loading from '../components/Loading';
import { useToast } from '../context/ToastContext';

const PROVIDER_LABELS = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  google: 'Google',
  deepseek: 'DeepSeek',
  mistral: 'Mistral',
};

const TUNABLE_LABELS = {
  fallback_model: 'Fallback model',
  default_temperature: 'Default temperature',
  elo_k_factor: 'ELO K-factor',
  rate_limit_snooze_seconds: 'Rate-limit snooze (seconds)',
  game_runner_pacing_seconds: 'Game runner pacing (seconds)',
};

const TUNABLE_HELPERS = {
  fallback_model:
    'Used when a model fails to initialise. Must match a key in models.yaml.',
  default_temperature: '0 = deterministic, 1 = creative.',
  elo_k_factor: 'Higher = ratings move faster after each game.',
  rate_limit_snooze_seconds:
    'How long a game stays paused after a 429 from the LLM provider.',
  game_runner_pacing_seconds:
    'Delay between AI turns in the background runner. Lower = faster tournaments.',
};

const Settings = () => {
  const toast = useToast();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  // Pending edits keyed by section.
  const [pendingKeys, setPendingKeys] = useState({});
  const [pendingTunables, setPendingTunables] = useState({});
  // Which keys are currently in plaintext-edit mode.
  const [editing, setEditing] = useState(new Set());

  const load = async () => {
    setLoading(true);
    try {
      setData(await getSettings());
      setPendingKeys({});
      setPendingTunables({});
      setEditing(new Set());
    } catch (e) {
      // Interceptor surfaces the toast.
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSave = async () => {
    if (!Object.keys(pendingKeys).length && !Object.keys(pendingTunables).length) {
      toast.info('No changes to save');
      return;
    }
    setSaving(true);
    try {
      // Coerce numeric tunables before sending.
      const numericTunables = {};
      for (const [k, v] of Object.entries(pendingTunables)) {
        const isNumber = typeof data?.tunables?.[k]?.default === 'number';
        numericTunables[k] = isNumber ? Number(v) : v;
      }
      await patchSettings({
        api_keys: pendingKeys,
        tunables: numericTunables,
      });
      toast.success('Settings saved');
      await load();
    } catch (e) {
      // Interceptor surfaces the toast.
    } finally {
      setSaving(false);
    }
  };

  const handleClearKey = async (provider) => {
    try {
      await clearApiKey(provider);
      toast.success(`${PROVIDER_LABELS[provider]} key cleared`);
      await load();
    } catch (e) {
      // Interceptor surfaces the toast.
    }
  };

  const handleClearTunable = async (key) => {
    try {
      await clearTunable(key);
      toast.success(`${TUNABLE_LABELS[key]} reset to default`);
      await load();
    } catch (e) {
      // Interceptor surfaces the toast.
    }
  };

  const toggleEditing = (provider) => {
    setEditing((prev) => {
      const next = new Set(prev);
      if (next.has(provider)) next.delete(provider);
      else next.add(provider);
      return next;
    });
  };

  if (loading) return <Loading message="Loading settings…" />;
  if (!data) return null;

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Sliders className="text-brand-600 dark:text-brand-500" size={28} />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Settings</h1>
        </div>
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="inline-flex items-center gap-2 px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white text-sm font-medium rounded-lg disabled:opacity-50"
        >
          <Save size={16} /> {saving ? 'Saving…' : 'Save changes'}
        </button>
      </div>

      {/* API Keys */}
      <section className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm">
        <header className="px-6 py-4 border-b border-gray-100 dark:border-gray-800 flex items-center gap-2">
          <Key size={18} className="text-brand-600 dark:text-brand-400" />
          <h2 className="font-semibold text-gray-900 dark:text-white">API Keys</h2>
        </header>
        <div className="p-6 space-y-4">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Keys stored here override the corresponding environment variables.
            Saved values are masked on read; the full key never leaves the
            server unencrypted.
          </p>
          {data.providers.map((provider) => {
            const info = data.api_keys[provider];
            const isEditing = editing.has(provider);
            const pending = pendingKeys[provider];
            return (
              <div
                key={provider}
                className="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700"
              >
                <div className="w-24 shrink-0 text-sm font-medium dark:text-gray-200">
                  {PROVIDER_LABELS[provider] ?? provider}
                </div>
                <div className="flex-1">
                  {isEditing ? (
                    <input
                      type="text"
                      autoFocus
                      placeholder="Paste new key"
                      value={pending ?? ''}
                      onChange={(e) =>
                        setPendingKeys({ ...pendingKeys, [provider]: e.target.value })
                      }
                      className="w-full px-2 py-1 rounded border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 text-sm font-mono"
                    />
                  ) : (
                    <div className="flex items-center gap-2 text-sm">
                      <span className="font-mono text-gray-700 dark:text-gray-300">
                        {info.preview ?? '— not set —'}
                      </span>
                      {info.set && (
                        <span
                          className={`text-[10px] uppercase px-1.5 py-0.5 rounded-full ${
                            info.source === 'override'
                              ? 'bg-brand-100 text-brand-700 dark:bg-brand-900/30 dark:text-brand-300'
                              : 'bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-300'
                          }`}
                        >
                          {info.source}
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <div className="flex gap-1 shrink-0">
                  <button
                    type="button"
                    onClick={() => toggleEditing(provider)}
                    className="p-1.5 text-gray-500 hover:bg-gray-200 dark:hover:bg-gray-700 rounded"
                    aria-label={isEditing ? 'Cancel' : 'Edit'}
                  >
                    {isEditing ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                  {info.source === 'override' && !isEditing && (
                    <button
                      type="button"
                      onClick={() => handleClearKey(provider)}
                      className="p-1.5 text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded"
                      aria-label="Clear override"
                    >
                      <X size={14} />
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Tunables */}
      <section className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm">
        <header className="px-6 py-4 border-b border-gray-100 dark:border-gray-800 flex items-center gap-2">
          <Sliders size={18} className="text-brand-600 dark:text-brand-400" />
          <h2 className="font-semibold text-gray-900 dark:text-white">Engine settings</h2>
        </header>
        <div className="p-6 space-y-4">
          {data.editable_tunables.map((key) => {
            const info = data.tunables[key];
            const pending = pendingTunables[key];
            const value = pending ?? info.value;
            return (
              <div key={key} className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium dark:text-gray-200">
                    {TUNABLE_LABELS[key] ?? key}
                  </label>
                  {info.overridden && (
                    <button
                      type="button"
                      onClick={() => handleClearTunable(key)}
                      className="inline-flex items-center gap-1 text-xs text-red-500 hover:text-red-600"
                    >
                      <RefreshCw size={12} /> reset (default {String(info.default)})
                    </button>
                  )}
                </div>
                <input
                  type="text"
                  value={value}
                  onChange={(e) =>
                    setPendingTunables({ ...pendingTunables, [key]: e.target.value })
                  }
                  className="w-full px-3 py-2 rounded border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-sm font-mono"
                />
                {TUNABLE_HELPERS[key] && (
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {TUNABLE_HELPERS[key]}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
};

export default Settings;
