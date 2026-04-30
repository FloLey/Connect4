import { http, HttpResponse } from 'msw';

const BASE = 'http://localhost:8000';

export const recordedRequests = {
  list: [],
  reset() {
    this.list = [];
  },
  push(method, url, body) {
    this.list.push({ method, url, body });
  },
};

const record = (method) => async ({ request }) => {
  let body = null;
  try {
    body = await request.clone().json();
  } catch (_) {
    body = null;
  }
  recordedRequests.push(method, request.url, body);
};

// Default handlers covering every method in src/api/client.js. Tests can
// override individual endpoints via server.use(...).
export const handlers = [
  http.get(`${BASE}/models`, () =>
    HttpResponse.json([
      { id: 'gpt-4o', provider: 'openai', label: 'GPT-4o' },
      { id: 'claude-3', provider: 'anthropic', label: 'Claude 3' },
    ])
  ),

  http.post(`${BASE}/games`, async (info) => {
    await record('POST')(info);
    return HttpResponse.json({
      id: 1,
      status: 'IN_PROGRESS',
      winner: null,
      history: [],
      created_at: new Date().toISOString(),
      player_1_type: 'human',
      player_2_type: 'human',
      player_1_token: 'token-1',
      player_2_token: 'token-2',
    });
  }),

  http.get(`${BASE}/games/history`, () => HttpResponse.json([])),
  http.get(`${BASE}/games/pending-human`, () => HttpResponse.json([])),
  http.get(`${BASE}/games/:id`, ({ params }) =>
    HttpResponse.json({
      id: Number(params.id),
      status: 'IN_PROGRESS',
      winner: null,
      history: [],
      created_at: new Date().toISOString(),
      player_1_type: 'human',
      player_2_type: 'human',
    })
  ),

  http.get(`${BASE}/stats/leaderboard`, () => HttpResponse.json([])),
  http.get(`${BASE}/stats/matrix`, () =>
    HttpResponse.json({ models: [], grid: {} })
  ),
  http.get(`${BASE}/stats/active-games`, () => HttpResponse.json([])),
  http.get(`${BASE}/stats/history`, () => HttpResponse.json([])),
  http.get(`${BASE}/stats/history-plot`, () => HttpResponse.json([])),

  http.get(`${BASE}/admin/status`, () =>
    HttpResponse.json({ games: 0, elo_ratings: 0, elo_history: 0 })
  ),
  http.delete(`${BASE}/admin/reset`, async (info) => {
    await record('DELETE')(info);
    return HttpResponse.json({ message: 'Database successfully wiped.' });
  }),

  http.post(`${BASE}/tournament/create`, async (info) => {
    await record('POST')(info);
    return HttpResponse.json({ id: 1, total_matches: 6, status: 'SETUP' });
  }),
  http.post(`${BASE}/tournament/create-evaluation`, async (info) => {
    await record('POST')(info);
    return HttpResponse.json({ id: 2, total_matches: 4, status: 'SETUP' });
  }),
  http.post(`${BASE}/tournament/:id/start`, () =>
    HttpResponse.json({ message: 'Tournament started' })
  ),
  http.post(`${BASE}/tournament/:id/stop`, () =>
    HttpResponse.json({ message: 'Tournament stopped' })
  ),
  http.post(`${BASE}/tournament/:id/pause`, () =>
    HttpResponse.json({ message: 'Tournament paused' })
  ),
  http.post(`${BASE}/tournament/:id/resume`, () =>
    HttpResponse.json({ message: 'Tournament resumed' })
  ),
  http.patch(`${BASE}/tournament/:id/config`, async (info) => {
    await record('PATCH')(info);
    return HttpResponse.json({ message: 'Tournament configuration updated' });
  }),
  http.get(`${BASE}/tournament/current`, () => HttpResponse.json(null)),

  // Settings (Tier 5)
  http.get(`${BASE}/settings`, () =>
    HttpResponse.json({
      providers: ['openai', 'anthropic', 'google', 'deepseek', 'mistral'],
      editable_tunables: [
        'default_temperature',
        'elo_k_factor',
        'fallback_model',
        'game_runner_pacing_seconds',
        'rate_limit_snooze_seconds',
      ],
      api_keys: {
        openai: { set: true, source: 'env', preview: '****ABCD' },
        anthropic: { set: false, source: null, preview: null },
        google: { set: false, source: null, preview: null },
        deepseek: { set: false, source: null, preview: null },
        mistral: { set: true, source: 'override', preview: '****WXYZ' },
      },
      tunables: {
        default_temperature: { value: 0.2, default: 0.2, overridden: false },
        elo_k_factor: { value: 32, default: 32, overridden: false },
        fallback_model: { value: 'gpt-4o', default: 'gpt-4o', overridden: false },
        game_runner_pacing_seconds: { value: 1.5, default: 1.5, overridden: false },
        rate_limit_snooze_seconds: { value: 600, default: 600, overridden: false },
      },
    })
  ),
  http.patch(`${BASE}/settings`, async (info) => {
    await record('PATCH')(info);
    return HttpResponse.json({ ok: true });
  }),
  http.delete(`${BASE}/settings/api-keys/:provider`, async (info) => {
    await record('DELETE')(info);
    return HttpResponse.json({ ok: true });
  }),
  http.delete(`${BASE}/settings/tunables/:key`, async (info) => {
    await record('DELETE')(info);
    return HttpResponse.json({ ok: true });
  }),
];
