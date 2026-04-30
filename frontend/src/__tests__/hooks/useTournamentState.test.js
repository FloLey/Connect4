import { describe, expect, it } from 'vitest';

import { __testing } from '../../hooks/useTournamentState';

const { reducer, initialState } = __testing;

const seedState = (overrides = {}) => ({
  ...initialState,
  ...overrides,
  dismissedIds: new Set(overrides.dismissedIds ?? []),
});

const inProgress = { id: 7, status: 'IN_PROGRESS', config: { concurrency: 2 } };
const completed = { id: 7, status: 'COMPLETED', config: { concurrency: 2 } };
const stopped = { id: 7, status: 'STOPPED', config: { concurrency: 2 } };

describe('useTournamentState reducer', () => {
  it('starts in loading phase with no tournament', () => {
    expect(initialState.phase).toBe('loading');
    expect(initialState.tournament).toBeNull();
  });

  it('POLL with null tournament moves to setup', () => {
    const next = reducer(seedState({ phase: 'live', tournament: inProgress }), {
      type: 'POLL',
      tournament: null,
      activeGames: [],
    });
    expect(next.phase).toBe('setup');
    expect(next.tournament).toBeNull();
  });

  it('POLL with IN_PROGRESS tournament moves to live', () => {
    const next = reducer(seedState(), {
      type: 'POLL',
      tournament: inProgress,
      activeGames: [{ id: 1 }],
    });
    expect(next.phase).toBe('live');
    expect(next.tournament).toEqual(inProgress);
    expect(next.activeGames).toEqual([{ id: 1 }]);
  });

  it('POLL with terminal status moves to completion when already watching it', () => {
    const next = reducer(seedState({ phase: 'live', tournament: inProgress }), {
      type: 'POLL',
      tournament: completed,
      activeGames: [],
    });
    expect(next.phase).toBe('completion');
    expect(next.tournament).toEqual(completed);
  });

  it('POLL with terminal status is ignored when not watching it', () => {
    const next = reducer(seedState({ phase: 'setup' }), {
      type: 'POLL',
      tournament: stopped,
      activeGames: [],
    });
    expect(next.phase).toBe('setup');
    expect(next.tournament).toBeNull();
  });

  it('POLL ignores tournaments in dismissedIds', () => {
    const state = seedState({ dismissedIds: [7] });
    const next = reducer(state, {
      type: 'POLL',
      tournament: inProgress,
      activeGames: [],
    });
    expect(next).toBe(state); // identity → no change
  });

  it('CREATE_OK transitions to live and clears dismissal for that id', () => {
    const next = reducer(seedState({ dismissedIds: [7] }), {
      type: 'CREATE_OK',
      tournament: inProgress,
    });
    expect(next.phase).toBe('live');
    expect(next.tournament).toEqual(inProgress);
    expect(next.dismissedIds.has(7)).toBe(false);
  });

  it('DISMISS adds the current tournament id to dismissedIds and returns to setup', () => {
    const next = reducer(seedState({ phase: 'live', tournament: inProgress }), {
      type: 'DISMISS',
    });
    expect(next.phase).toBe('setup');
    expect(next.tournament).toBeNull();
    expect(next.dismissedIds.has(7)).toBe(true);
  });

  it('OPTIMISTIC_STATUS updates only the status field', () => {
    const next = reducer(seedState({ phase: 'live', tournament: inProgress }), {
      type: 'OPTIMISTIC_STATUS',
      status: 'PAUSED',
    });
    expect(next.tournament.status).toBe('PAUSED');
    expect(next.tournament.config).toEqual({ concurrency: 2 });
  });

  it('OPTIMISTIC_CONFIG merges into config without dropping other keys', () => {
    const tournament = { ...inProgress, config: { concurrency: 2, rounds: 3 } };
    const next = reducer(seedState({ phase: 'live', tournament }), {
      type: 'OPTIMISTIC_CONFIG',
      config: { concurrency: 7 },
    });
    expect(next.tournament.config).toEqual({ concurrency: 7, rounds: 3 });
  });

  it('GAME_STARTED appends to activeGames in stable id-sorted order', () => {
    const state = seedState({
      phase: 'live',
      tournament: inProgress,
      activeGames: [{ id: 5 }, { id: 9 }],
    });
    const next = reducer(state, {
      type: 'GAME_STARTED',
      game: { id: 7, player_1: 'a', player_2: 'b' },
    });
    expect(next.activeGames.map((g) => g.id)).toEqual([5, 7, 9]);
  });

  it('GAME_STARTED is a no-op when the game is already in the list', () => {
    const state = seedState({
      phase: 'live',
      tournament: inProgress,
      activeGames: [{ id: 7 }],
    });
    const next = reducer(state, { type: 'GAME_STARTED', game: { id: 7 } });
    expect(next).toBe(state);
  });

  it('GAME_COMPLETED removes the game from activeGames', () => {
    const state = seedState({
      phase: 'live',
      tournament: inProgress,
      activeGames: [{ id: 1 }, { id: 2 }, { id: 3 }],
    });
    const next = reducer(state, { type: 'GAME_COMPLETED', gameId: 2 });
    expect(next.activeGames.map((g) => g.id)).toEqual([1, 3]);
  });

  it('RESET_FOR_DB_SWITCH wipes everything except availableModels', () => {
    const state = seedState({
      phase: 'live',
      tournament: inProgress,
      availableModels: [{ id: 'a' }],
      dismissedIds: [99],
    });
    const next = reducer(state, { type: 'RESET_FOR_DB_SWITCH' });
    expect(next.phase).toBe('loading');
    expect(next.tournament).toBeNull();
    expect(next.availableModels).toEqual([{ id: 'a' }]);
    expect(next.dismissedIds.size).toBe(0);
  });
});
