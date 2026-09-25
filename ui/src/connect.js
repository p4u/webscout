// "Connect AI tools": how to point an MCP client at this server. The URL is
// this page's own origin plus /mcp (nginx proxies it to the api), so what is
// shown is always the address the person is already using.
//
// The token: a password-protected server hands it to a logged-in person in
// `GET /api/mcp` (`token`), and the snippets are filled with it — masked on
// screen until revealed, real on the clipboard. An open server never sends it;
// a person may paste it to fill in the snippets, and it stays in this page only.
// Either way it is never stored or logged.

const TOKEN_PLACEHOLDER = 'YOUR_TOKEN';
const TOKEN_MASK = '•'.repeat(16);

/** One entry per client. `where` says which file or command; `code` builds the snippet. */
export const CLIENTS = [
  {
    id: 'claude',
    name: 'Claude Code',
    steps: [
      {
        where: 'Run in a terminal. Add --scope user to make it available in every project.',
        code: (url, token) =>
          `claude mcp add --transport http webscout ${url} \\\n  --header "Authorization: Bearer ${token}"`,
      },
      {
        where: 'Check it connected, then ask Claude to "use webscout to find …".',
        code: () => 'claude mcp list',
      },
    ],
  },
  {
    id: 'opencode',
    name: 'opencode',
    steps: [
      {
        where:
          'Add to opencode.json in your project, or to ~/.config/opencode/opencode.json for every project. The timeout matters: opencode gives up on a tool call after 5 s by default, and each check waits about 10 s.',
        code: (url, token) =>
          JSON.stringify(
            {
              $schema: 'https://opencode.ai/config.json',
              mcp: {
                webscout: {
                  type: 'remote',
                  url,
                  headers: { Authorization: `Bearer ${token}` },
                  oauth: false,
                  timeout: 90000,
                  enabled: true,
                },
              },
            },
            null,
            2,
          ),
      },
      { where: 'Check it connected:', code: () => 'opencode mcp list' },
    ],
  },
  {
    id: 'pi',
    name: 'pi',
    steps: [
      {
        where: 'pi talks to MCP servers through the pi-mcp-adapter package. Install it once:',
        code: () => 'pi install npm:pi-mcp-adapter',
      },
      {
        where: 'Then add webscout to ~/.pi/agent/mcp.json (or .pi/mcp.json in a project), and run /mcp inside pi to see it.',
        code: (url, token) =>
          JSON.stringify(
            {
              mcpServers: {
                webscout: {
                  url,
                  headers: { Authorization: `Bearer ${token}` },
                  directTools: true,
                },
              },
            },
            null,
            2,
          ),
      },
    ],
  },
  {
    id: 'others',
    name: 'Others',
    steps: [
      {
        where:
          'Any client that speaks MCP over HTTP ("Streamable HTTP"): give it this URL and the header below.',
        code: (url, token) => `URL:    ${url}\nHeader: Authorization: Bearer ${token}`,
      },
      {
        where: 'Cursor — ~/.cursor/mcp.json',
        code: (url, token) =>
          JSON.stringify(
            { mcpServers: { webscout: { url, headers: { Authorization: `Bearer ${token}` } } } },
            null,
            2,
          ),
      },
      {
        where: 'VS Code (Copilot) — .vscode/mcp.json',
        code: (url, token) =>
          JSON.stringify(
            {
              servers: {
                webscout: { type: 'http', url, headers: { Authorization: `Bearer ${token}` } },
              },
            },
            null,
            2,
          ),
      },
      {
        where: 'Codex CLI — ~/.codex/config.toml (the token is read from the WEBSCOUT_MCP_TOKEN variable)',
        code: (url) =>
          `[mcp_servers.webscout]\nurl = "${url}"\nbearer_token_env_var = "WEBSCOUT_MCP_TOKEN"`,
      },
      {
        where: 'Gemini CLI — ~/.gemini/settings.json',
        code: (url, token) =>
          JSON.stringify(
            {
              mcpServers: {
                webscout: { httpUrl: url, headers: { Authorization: `Bearer ${token}` }, timeout: 90000 },
              },
            },
            null,
            2,
          ),
      },
      {
        where:
          'Clients that only run local (stdio) servers, such as Claude Desktop: bridge with mcp-remote (needs Node.js).',
        code: (url, token) =>
          JSON.stringify(
            {
              mcpServers: {
                webscout: {
                  command: 'npx',
                  args: [
                    '-y',
                    'mcp-remote',
                    url,
                    '--header',
                    'Authorization:${AUTH_HEADER}',
                    ...(url.startsWith('http://') && !/\/\/(localhost|127\.0\.0\.1)[:/]/.test(url)
                      ? ['--allow-http']
                      : []),
                  ],
                  env: { AUTH_HEADER: `Bearer ${token}` },
                },
              },
            },
            null,
            2,
          ),
      },
    ],
  },
];

export const TOOL_NOTES = [
  ['search', 'runs a search with the same options as this page and usually returns the verified answer in the same call; a longer search returns a request id instead.'],
  ['start_search', 'starts a search in the background and returns its request id at once.'],
  ['get_search_result', 'waits about 10 s for the result; the assistant calls it again while the search is still running. "simple" gives the result text, "full" adds sources, notes, cost and statistics.'],
  ['get_search_status', 'shows progress, pages read and cost so far.'],
  ['cancel_search, list_searches, list_search_options', 'stop a search, find a request id again, and describe every option.'],
];

export function mcpUrl() {
  return `${location.origin}/mcp`;
}

export function tokenOrPlaceholder(raw) {
  const t = String(raw ?? '').trim();
  return t || TOKEN_PLACEHOLDER;
}

/** What the snippets show for a server token that has not been revealed. */
export function maskedToken() {
  return TOKEN_MASK;
}
