import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getModels = async () => {
  const response = await apiClient.get('/models');
  return response.data;
};

export const createGame = async (player1, player2) => {
  const response = await apiClient.post('/games', {
    player_1: player1,
    player_2: player2,
  });
  return response.data;
};

export const getGame = async (gameId) => {
  const response = await apiClient.get(`/games/${gameId}`);
  return response.data;
};

export const getLeaderboard = async () => {
  const response = await apiClient.get('/stats/leaderboard');
  return response.data;
};

// --- NEW ---
export const getMatrix = async () => {
  const response = await apiClient.get('/stats/matrix');
  return response.data;
};
// -----------

export const getActiveGames = async () => {
  const response = await apiClient.get('/stats/active-games');
  return response.data;
};

export const getHistory = async (modelName) => {
  const url = modelName ? `/stats/history?model=${modelName}` : '/stats/history';
  const response = await apiClient.get(url);
  return response.data;
};

export const getHistoryPlot = async () => {
  const response = await apiClient.get('/stats/history-plot');
  return response.data;
};

// Admin API functions
export const getAdminStatus = async () => {
  const response = await apiClient.get('/admin/status');
  return response.data;
};

export const resetDatabase = async () => {
  const response = await apiClient.delete('/admin/reset', {
    params: { confirmation: 'I-UNDERSTAND-THIS-DELETES-EVERYTHING' }
  });
  return response.data;
};

export const getGameHistory = async (skip = 0, limit = 50) => {
  const response = await apiClient.get('/games/history', { 
    params: { skip, limit } 
  });
  return response.data;
};

export const getPendingHumanGames = async () => {
  const response = await apiClient.get('/games/pending-human');
  return response.data; // Returns array of IDs, e.g., [1, 5, 8]
};

// --- TOURNAMENT API ---
export const createTournament = async (models, rounds, concurrency) => {
  const response = await apiClient.post('/tournament/create', {
    models,
    rounds,
    concurrency
  });
  return response.data;
};

export const startTournament = async (tournamentId) => {
  const response = await apiClient.post(`/tournament/${tournamentId}/start`);
  return response.data;
};

export const stopTournament = async (tournamentId) => {
  const response = await apiClient.post(`/tournament/${tournamentId}/stop`);
  return response.data;
};

export const getCurrentTournament = async () => {
  const response = await apiClient.get('/tournament/current');
  return response.data; // Returns null if no tournament exists
};

// --- NEW: Tournament Pause/Resume/Config API ---
export const pauseTournament = async (tournamentId) => {
  const response = await apiClient.post(`/tournament/${tournamentId}/pause`);
  return response.data;
};

export const resumeTournament = async (tournamentId) => {
  const response = await apiClient.post(`/tournament/${tournamentId}/resume`);
  return response.data;
};

export const updateTournamentConfig = async (tournamentId, concurrency) => {
  const response = await apiClient.patch(`/tournament/${tournamentId}/config`, {
    concurrency
  });
  return response.data;
};

export const createEvaluationTournament = async (target, benchmarks, rounds, concurrency) => {
  const response = await apiClient.post('/tournament/create-evaluation', {
    target_model: target,
    benchmark_models: benchmarks,
    rounds,
    concurrency
  });
  return response.data;
};
// ----------------------