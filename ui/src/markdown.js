// Markdown rendering for content scraped from the web. Treat every byte as hostile:
// marked produces HTML, DOMPurify is the only thing standing between it and the DOM.

import { Marked } from 'marked';
import DOMPurify from 'dompurify';

const marked = new Marked({ gfm: true, breaks: false });

// Links open in a new tab; noopener/noreferrer so the target can never touch us.
DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A') {
    const href = node.getAttribute('href') || '';
    if (href) {
      node.setAttribute('target', '_blank');
      node.setAttribute('rel', 'noopener noreferrer');
    }
  }
});

const PURIFY_CONFIG = {
  USE_PROFILES: { html: true },
  // No inline event handlers, no styles, no form controls in rendered output.
  FORBID_TAGS: ['style', 'form', 'input', 'button', 'iframe', 'object', 'embed'],
  FORBID_ATTR: ['style', 'srcset', 'formaction', 'ping'],
  ADD_ATTR: ['target', 'rel'],
};

/** Render markdown to sanitised HTML. Never throws. */
export function renderMarkdown(source) {
  const text = typeof source === 'string' ? source : '';
  let html;
  try {
    html = marked.parse(text);
  } catch {
    // A parser failure must not blank the page: fall back to escaped plain text.
    html = `<pre><code>${escapeHtml(text)}</code></pre>`;
  }
  return DOMPurify.sanitize(html, PURIFY_CONFIG);
}

/** Render non-markdown payloads (json, csv, jsonl) as an escaped code block. */
export function renderPlain(source, language = '') {
  const cls = language ? ` class="language-${escapeHtml(language)}"` : '';
  return DOMPurify.sanitize(
    `<pre><code${cls}>${escapeHtml(typeof source === 'string' ? source : '')}</code></pre>`,
    PURIFY_CONFIG,
  );
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

/** Wrap tables so they scroll horizontally instead of blowing out the layout. */
export function wrapTables(container) {
  for (const table of container.querySelectorAll('table')) {
    if (table.parentElement?.classList.contains('table-wrap')) continue;
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    table.replaceWith(wrap);
    wrap.appendChild(table);
  }
}
