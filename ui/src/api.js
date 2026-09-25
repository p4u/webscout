// Thin client for the webscout API. Same-origin in production (nginx proxies /api/),
// proxied by Vite in development. Every request sends the session cookie
// (`credentials: 'same-origin'`); a 401 surfaces as an ApiError with status 401,
// which the app answers by showing the login card.

export class ApiError extends Error {
  constructor(message, { status = 0, cause } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    if (cause) this.cause = cause;
  }
}

async function readErrorBody(res) {
  // The API returns JSON errors; a proxy or crash may return HTML or nothing.
  let body = '';
  try {
    body = await res.text();
  } catch {
    return '';
  }
  if (!body) return '';
  try {
    const parsed = JSON.parse(body);
    for (const key of ['message', 'error', 'detail']) {
      const value = parsed?.[key];
      if (typeof value === 'string' && value) return value;
      if (value && typeof value === 'object') return JSON.stringify(value);
    }
    return JSON.stringify(parsed);
  } catch {
    return body.slice(0, 400).trim();
  }
}

export async function fetchOptions(signal) {
  let res;
  try {
    res = await fetch('/api/options', {
      headers: { accept: 'application/json' },
      credentials: 'same-origin',
      signal,
    });
  } catch (err) {
    throw new ApiError('Could not reach the webscout API. Is it running?', { cause: err });
  }
  if (!res.ok) {
    throw new ApiError(
      (await readErrorBody(res)) || `The API returned HTTP ${res.status} for /api/options.`,
      { status: res.status },
    );
  }
  let schema;
  try {
    schema = await res.json();
  } catch (err) {
    throw new ApiError('The API returned something that is not JSON for /api/options.', {
      cause: err,
    });
  }
  if (!schema || !Array.isArray(schema.groups)) {
    throw new ApiError('The /api/options response has no "groups" array.');
  }
  return schema;
}

/**
 * GET /api/session → `{ auth_required, authenticated }`.
 *
 * An older server has no such endpoint (404) and never asked for a login, so
 * any failure to get an answer reads as "open": the app boots normally, and if
 * the API is really down the options request says so in its own words.
 */
export async function fetchSession() {
  const open = { auth_required: false, authenticated: true };
  try {
    const res = await fetch('/api/session', {
      headers: { accept: 'application/json' },
      credentials: 'same-origin',
      signal: AbortSignal.timeout?.(8000),
    });
    if (!res.ok) return open;
    const body = await res.json();
    return {
      auth_required: body?.auth_required === true,
      authenticated: body?.authenticated === true,
    };
  } catch {
    return open;
  }
}

/** POST /api/login. Resolves on success; throws ApiError (401 wrong, 429 throttled). */
export async function login(password) {
  let res;
  try {
    res = await fetch('/api/login', {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/json' },
      body: JSON.stringify({ password }),
      credentials: 'same-origin',
    });
  } catch (err) {
    throw new ApiError('Could not reach the server. Check your connection and try again.', {
      cause: err,
    });
  }
  if (!res.ok) {
    throw new ApiError((await readErrorBody(res)) || `The server returned HTTP ${res.status}.`, {
      status: res.status,
    });
  }
}

/** POST /api/logout. Best effort: the card is shown whatever the answer. */
export async function logout() {
  try {
    await fetch('/api/logout', { method: 'POST', credentials: 'same-origin' });
  } catch {
    /* the cookie may outlive a failed request; the next 401 brings the card back */
  }
}

/** GET /api/mcp: whether MCP is on, and (logged in, password set) its token. */
export async function fetchMcpInfo() {
  let res;
  try {
    res = await fetch('/api/mcp', {
      headers: { accept: 'application/json' },
      credentials: 'same-origin',
    });
  } catch (err) {
    throw new ApiError('Could not reach the server.', { cause: err });
  }
  if (!res.ok) {
    throw new ApiError((await readErrorBody(res)) || `HTTP ${res.status}`, { status: res.status });
  }
  return res.json();
}

export function downloadUrl(runId, format) {
  return `/api/runs/${encodeURIComponent(runId)}/download?format=${encodeURIComponent(format)}`;
}

/**
 * POST /api/search and yield each NDJSON event as it arrives.
 *
 * Partial lines are buffered across chunks; a malformed line is surfaced as a
 * synthetic error event rather than silently dropped.
 */
export async function* streamSearch({ query, options, signal }) {
  let res;
  try {
    res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/x-ndjson' },
      body: JSON.stringify({ query, options }),
      credentials: 'same-origin',
      signal,
    });
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    throw new ApiError('Could not reach the webscout API. Is it running?', { cause: err });
  }

  if (!res.ok) {
    throw new ApiError(
      (await readErrorBody(res)) || `The API returned HTTP ${res.status}.`,
      { status: res.status },
    );
  }
  if (!res.body) {
    throw new ApiError('The API returned no response body to stream.');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let newline;
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        const event = parseLine(line);
        if (event) yield event;
      }
    }
    buffer += decoder.decode();
    const tail = parseLine(buffer);
    if (tail) yield tail;
  } finally {
    // Cancelling the reader lets the server observe the disconnect and abort the run.
    try {
      await reader.cancel();
    } catch {
      /* already closed */
    }
  }
}

function parseLine(raw) {
  const line = raw.trim();
  if (!line) return null;
  try {
    const event = JSON.parse(line);
    if (!event || typeof event !== 'object' || typeof event.type !== 'string') {
      return { type: 'error', message: 'The API sent an event with no "type" field.' };
    }
    return event;
  } catch {
    return {
      type: 'error',
      message: `The API sent a line that is not valid JSON: ${line.slice(0, 160)}`,
    };
  }
}
