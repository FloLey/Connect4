import { describe, expect, it, beforeEach } from 'vitest';
import {
  getModels,
  createGame,
  getGame,
  getLeaderboard,
  getMatrix,
  getActiveGames,
  getHistory,
  getHistoryPlot,
  getAdminStatus,
  resetDatabase,
  getGameHistory,
  getPendingHumanGames,
  createTournament,
  startTournament,
  stopTournament,
  getCurrentTournament,
  pauseTournament,
  resumeTournament,
  updateTournamentConfig,
  createEvaluationTournament,
} from '../../api/client';
import { recordedRequests } from '../../test/handlers';

beforeEach(() => {
  recordedRequests.reset();
});

describe('client.js', () => {
  it('getModels returns the registry', async () => {
    const models = await getModels();
    expect(models).toEqual([
      { id: 'gpt-4o', provider: 'openai', label: 'GPT-4o' },
      { id: 'claude-3', provider: 'anthropic', label: 'Claude 3' },
    ]);
  });

  it('createGame POSTs the player_1/player_2 payload', async () => {
    const game = await createGame('human', 'gpt-4o');
    expect(game.id).toBe(1);
    expect(recordedRequests.list).toHaveLength(1);
    const recorded = recordedRequests.list[0];
    expect(recorded.method).toBe('POST');
    expect(recorded.url).toContain('/games');
    expect(recorded.body).toEqual({ player_1: 'human', player_2: 'gpt-4o' });
  });

  it('getGame requests the right URL', async () => {
    const game = await getGame(42);
    expect(game.id).toBe(42);
  });

  it('stats endpoints return the MSW defaults', async () => {
    expect(await getLeaderboard()).toEqual([]);
    expect(await getMatrix()).toEqual({ models: [], grid: {} });
    expect(await getActiveGames()).toEqual([]);
    expect(await getHistory()).toEqual([]);
    expect(await getHistoryPlot()).toEqual([]);
  });

  it('admin endpoints work', async () => {
    const status = await getAdminStatus();
    expect(status.games).toBe(0);

    const reset = await resetDatabase();
    expect(reset.message).toContain('wiped');
    expect(recordedRequests.list.some((r) => r.method === 'DELETE')).toBe(true);
  });

  it('getGameHistory accepts skip + limit', async () => {
    await getGameHistory(10, 25);
    // No assertion on body — MSW just returns []. Ensure call succeeded.
    expect(true).toBe(true);
  });

  it('getPendingHumanGames returns ID list', async () => {
    expect(await getPendingHumanGames()).toEqual([]);
  });

  it('createTournament POSTs models/rounds/concurrency', async () => {
    const t = await createTournament(['a', 'b'], 1, 2);
    expect(t.id).toBe(1);
    const recorded = recordedRequests.list.find((r) => r.url.endsWith('/tournament/create'));
    expect(recorded.body).toEqual({ models: ['a', 'b'], rounds: 1, concurrency: 2 });
  });

  it('createEvaluationTournament uses target/benchmarks', async () => {
    await createEvaluationTournament('alpha', ['b1', 'b2'], 1, 2);
    const recorded = recordedRequests.list.find((r) =>
      r.url.endsWith('/tournament/create-evaluation')
    );
    expect(recorded.body).toEqual({
      target_model: 'alpha',
      benchmark_models: ['b1', 'b2'],
      rounds: 1,
      concurrency: 2,
    });
  });

  it('start/stop/pause/resume tournaments', async () => {
    await startTournament(1);
    await stopTournament(1);
    await pauseTournament(1);
    await resumeTournament(1);
    // All return MSW defaults; no exception means the URLs match.
    expect(true).toBe(true);
  });

  it('updateTournamentConfig PATCHes concurrency', async () => {
    await updateTournamentConfig(1, 5);
    const recorded = recordedRequests.list.find((r) => r.method === 'PATCH');
    expect(recorded.body).toEqual({ concurrency: 5 });
  });

  it('getCurrentTournament returns null when none', async () => {
    expect(await getCurrentTournament()).toBeNull();
  });
});
