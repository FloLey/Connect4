import { useCallback, useEffect, useReducer } from 'react';

import {
  createEvaluationTournament,
  createTournament,
  getActiveGames,
  getCurrentTournament,
  getModels,
  pauseTournament,
  resumeTournament,
  startTournament,
  stopTournament,
  updateTournamentConfig,
} from '../api/client';
import { useDatabase } from '../context/DatabaseContext';
import { useToast } from '../context/ToastContext';
import { usePolling } from './usePolling';
import { useTournamentSocket } from './useTournamentSocket';

const POLL_INTERVAL_MS = 10000;

const initialState = {
  phase: 'loading', // 'loading' | 'setup' | 'live' | 'completion'
  tournament: null,
  activeGames: [],
  availableModels: [],
  dismissedIds: new Set(),
};

const isTerminal = (status) =>
  status === 'COMPLETED' || status === 'STOPPED';

function reducer(state, action) {
  switch (action.type) {
    case 'MODELS_LOADED':
      return { ...state, availableModels: action.models };

    case 'POLL': {
      const { tournament, activeGames } = action;

      if (!tournament) {
        // No active tournament in DB.
        return {
          ...state,
          phase: 'setup',
          tournament: null,
          activeGames: [],
        };
      }

      // Filter out tournaments the user explicitly dismissed.
      if (state.dismissedIds.has(tournament.id)) {
        return state;
      }

      // If the polled tournament is terminal AND we weren't already watching
      // it, treat it as old news and stay on setup.
      if (
        isTerminal(tournament.status) &&
        state.tournament?.id !== tournament.id
      ) {
        return state.phase === 'loading'
          ? { ...state, phase: 'setup' }
          : state;
      }

      const phase = isTerminal(tournament.status) ? 'completion' : 'live';
      return {
        ...state,
        phase,
        tournament,
        activeGames: activeGames ?? state.activeGames,
      };
    }

    case 'CREATE_OK':
      return {
        ...state,
        phase: 'live',
        tournament: action.tournament,
        activeGames: [],
        // A freshly-created tournament is one we definitely want to watch.
        dismissedIds: new Set(
          [...state.dismissedIds].filter((id) => id !== action.tournament.id)
        ),
      };

    case 'DISMISS': {
      if (!state.tournament) {
        return { ...state, phase: 'setup' };
      }
      const dismissedIds = new Set(state.dismissedIds);
      dismissedIds.add(state.tournament.id);
      return {
        ...state,
        phase: 'setup',
        tournament: null,
        activeGames: [],
        dismissedIds,
      };
    }

    case 'OPTIMISTIC_STATUS': {
      if (!state.tournament) return state;
      return {
        ...state,
        tournament: { ...state.tournament, status: action.status },
      };
    }

    case 'OPTIMISTIC_CONFIG': {
      if (!state.tournament) return state;
      return {
        ...state,
        tournament: {
          ...state.tournament,
          config: { ...(state.tournament.config || {}), ...action.config },
        },
      };
    }

    case 'GAME_STARTED': {
      // Push event from the tournament WS; merge if not already in the list.
      if (state.activeGames.some((g) => g.id === action.game.id)) return state;
      return {
        ...state,
        activeGames: stableSortById([...state.activeGames, action.game]),
      };
    }

    case 'GAME_COMPLETED':
      return {
        ...state,
        activeGames: state.activeGames.filter((g) => g.id !== action.gameId),
      };

    case 'RESET_FOR_DB_SWITCH':
      return { ...initialState, availableModels: state.availableModels };

    default:
      return state;
  }
}

const stableSortById = (games) => [...games].sort((a, b) => a.id - b.id);

export const useTournamentState = () => {
  const { dbEnv } = useDatabase();
  const toast = useToast();
  const [state, dispatch] = useReducer(reducer, initialState);

  // Reset on DB env switch.
  useEffect(() => {
    dispatch({ type: 'RESET_FOR_DB_SWITCH' });
  }, [dbEnv]);

  // Load model list once per dbEnv.
  useEffect(() => {
    let cancelled = false;
    getModels()
      .then((models) => {
        if (!cancelled) dispatch({ type: 'MODELS_LOADED', models });
      })
      .catch(() => {
        // Interceptor surfaces the toast.
      });
    return () => {
      cancelled = true;
    };
  }, [dbEnv]);

  // Poll for current tournament + live games. The 10s cadence is the safety
  // net — sub-second updates flow through the per-tournament WebSocket below.
  usePolling(
    async () => {
      const tournament = await getCurrentTournament();
      let activeGames = [];
      if (tournament && tournament.status === 'IN_PROGRESS') {
        activeGames = stableSortById(await getActiveGames());
      }
      dispatch({ type: 'POLL', tournament, activeGames });
    },
    POLL_INTERVAL_MS,
    { deps: [dbEnv] }
  );

  // Real-time control-plane events for the active tournament. The hook is a
  // no-op when tournamentId is falsy.
  useTournamentSocket(state.tournament?.id, (event) => {
    if (event.type === 'GAME_STARTED') {
      dispatch({ type: 'GAME_STARTED', game: event.game });
    } else if (event.type === 'GAME_COMPLETED') {
      dispatch({ type: 'GAME_COMPLETED', gameId: event.game_id });
    }
  });

  // -- actions --------------------------------------------------------------

  const create = useCallback(
    async ({ mode, models, rounds, concurrency, targetModel }) => {
      if (models.length < 2) {
        toast.error('Select at least 2 models');
        return;
      }
      if (mode === 'EVALUATION' && !targetModel) {
        toast.error('Select a target model for evaluation');
        return;
      }
      try {
        const tournament =
          mode === 'ROUND_ROBIN'
            ? await createTournament(models, rounds, concurrency)
            : await createEvaluationTournament(
                targetModel,
                models.filter((m) => m !== targetModel),
                rounds,
                concurrency
              );
        dispatch({ type: 'CREATE_OK', tournament });
      } catch (e) {
        // Interceptor surfaces the toast.
      }
    },
    [toast]
  );

  const start = useCallback(async () => {
    if (!state.tournament) return;
    try {
      await startTournament(state.tournament.id);
      dispatch({ type: 'OPTIMISTIC_STATUS', status: 'IN_PROGRESS' });
    } catch (e) {
      // toast via interceptor
    }
  }, [state.tournament]);

  const stop = useCallback(async () => {
    if (!state.tournament) return;
    if (
      typeof window !== 'undefined' &&
      !window.confirm('Stop the tournament? Running games will finish, pending games will remain pending.')
    ) {
      return;
    }
    try {
      await stopTournament(state.tournament.id);
      dispatch({ type: 'OPTIMISTIC_STATUS', status: 'STOPPED' });
    } catch (e) {
      // toast via interceptor
    }
  }, [state.tournament]);

  const pause = useCallback(async () => {
    if (!state.tournament) return;
    try {
      await pauseTournament(state.tournament.id);
      dispatch({ type: 'OPTIMISTIC_STATUS', status: 'PAUSED' });
    } catch (e) {
      // toast via interceptor
    }
  }, [state.tournament]);

  const resume = useCallback(async () => {
    if (!state.tournament) return;
    try {
      await resumeTournament(state.tournament.id);
      dispatch({ type: 'OPTIMISTIC_STATUS', status: 'IN_PROGRESS' });
    } catch (e) {
      // toast via interceptor
    }
  }, [state.tournament]);

  const updateConcurrency = useCallback(
    async (newConcurrency) => {
      if (!state.tournament) return;
      try {
        await updateTournamentConfig(state.tournament.id, newConcurrency);
        dispatch({ type: 'OPTIMISTIC_CONFIG', config: { concurrency: newConcurrency } });
        toast.success('Configuration updated');
      } catch (e) {
        // toast via interceptor
      }
    },
    [state.tournament, toast]
  );

  const dismiss = useCallback(() => {
    dispatch({ type: 'DISMISS' });
  }, []);

  return {
    phase: state.phase,
    tournament: state.tournament,
    activeGames: state.activeGames,
    availableModels: state.availableModels,
    actions: { create, start, stop, pause, resume, updateConcurrency, dismiss },
  };
};

// Exposed for unit tests — kept here to avoid leaking it elsewhere.
export const __testing = { reducer, initialState, isTerminal };
