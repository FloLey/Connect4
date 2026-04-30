import { describe, expect, it, beforeEach } from 'vitest';

import {
  cleanupOldGameTokens,
  saveGameTokenWithTimestamp,
  initTokenCleanup,
} from '../../utils/tokenCleanup';

const ONE_DAY_MS = 24 * 60 * 60 * 1000;

beforeEach(() => {
  localStorage.clear();
});

describe('saveGameTokenWithTimestamp', () => {
  it('stores both the token and a creation timestamp', () => {
    saveGameTokenWithTimestamp(42, 'tok-123');
    expect(localStorage.getItem('game_42_token')).toBe('tok-123');
    expect(Number(localStorage.getItem('game_42_created'))).toBeGreaterThan(0);
  });
});

describe('cleanupOldGameTokens', () => {
  it('removes tokens older than 24 hours', () => {
    const oldKey = 'game_1_token';
    const oldCreated = 'game_1_created';
    localStorage.setItem(oldKey, 'old-tok');
    localStorage.setItem(oldCreated, String(Date.now() - 2 * ONE_DAY_MS));

    const removed = cleanupOldGameTokens();

    expect(removed).toBe(1);
    expect(localStorage.getItem(oldKey)).toBeNull();
    expect(localStorage.getItem(oldCreated)).toBeNull();
  });

  it('keeps tokens younger than 24 hours', () => {
    saveGameTokenWithTimestamp(2, 'fresh');
    const removed = cleanupOldGameTokens();
    expect(removed).toBe(0);
    expect(localStorage.getItem('game_2_token')).toBe('fresh');
  });

  it('removes tokens with no creation timestamp (orphans)', () => {
    localStorage.setItem('game_3_token', 'orphan');
    const removed = cleanupOldGameTokens();
    expect(removed).toBe(1);
    expect(localStorage.getItem('game_3_token')).toBeNull();
  });

  it('ignores unrelated localStorage keys', () => {
    localStorage.setItem('not_a_token', 'leave me');
    saveGameTokenWithTimestamp(4, 'fresh');
    const removed = cleanupOldGameTokens();
    expect(removed).toBe(0);
    expect(localStorage.getItem('not_a_token')).toBe('leave me');
  });
});

describe('initTokenCleanup', () => {
  it('runs cleanup on first call and records the timestamp', () => {
    localStorage.setItem('game_5_token', 'orphan');
    initTokenCleanup();
    expect(localStorage.getItem('game_5_token')).toBeNull();
    expect(Number(localStorage.getItem('last_token_cleanup'))).toBeGreaterThan(0);
  });

  it('skips cleanup if already run within the last day', () => {
    localStorage.setItem('last_token_cleanup', String(Date.now() - 60_000));
    localStorage.setItem('game_6_token', 'orphan');
    initTokenCleanup();
    expect(localStorage.getItem('game_6_token')).toBe('orphan');
  });
});
