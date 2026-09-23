// Renders the control surface generically from GET /api/options. The UI must never
// hardcode the parameter list: everything here is driven by `type`, `default`,
// `min`, `max`, `values` and `help`.

const KNOWN_TYPES = new Set(['integer', 'number', 'boolean', 'enum', 'string']);

/**
 * @param {object} schema  the /api/options payload
 * @param {(name: string, value: unknown) => void} onChange
 */
export function createControls(schema, onChange) {
  const options = new Map(); // name -> option spec
  const values = new Map(); // name -> current value
  const nodes = new Map(); // name -> { input, label, chip }

  for (const group of schema.groups ?? []) {
    for (const option of group.options ?? []) {
      if (!option || typeof option.name !== 'string') continue;
      options.set(option.name, option);
      values.set(option.name, normalise(option, option.default));
    }
  }

  function isDirty(name) {
    const option = options.get(name);
    if (!option) return false;
    return !sameValue(values.get(name), normalise(option, option.default));
  }

  function markDirty(name) {
    const entry = nodes.get(name);
    if (!entry) return;
    const dirty = String(isDirty(name));
    entry.chip?.setAttribute('data-dirty', dirty);
    entry.label?.setAttribute('data-dirty', dirty);
  }

  function set(name, raw, { silent = false } = {}) {
    const option = options.get(name);
    if (!option) return;
    values.set(name, normalise(option, raw));
    markDirty(name);
    if (!silent) onChange(name, values.get(name));
  }

  function register(name, input, { chip, label, sync } = {}) {
    nodes.set(name, { input, chip, label, sync });
    markDirty(name);
  }

  /** Only the options the user actually changed. */
  function changed() {
    const out = {};
    for (const name of options.keys()) {
      if (isDirty(name)) out[name] = values.get(name);
    }
    return out;
  }

  function reset() {
    for (const [name, option] of options) {
      values.set(name, normalise(option, option.default));
      const entry = nodes.get(name);
      if (entry?.input) writeToInput(entry.input, option, values.get(name));
      entry?.sync?.();
      markDirty(name);
    }
  }

  function group(id) {
    return (schema.groups ?? []).find((g) => g.id === id);
  }

  return {
    options,
    changed,
    reset,
    changedCount: () => Object.keys(changed()).length,
    /** Compact chips for the always-visible basic row. */
    renderBasic(container, groupId = 'basic') {
      container.replaceChildren();
      for (const option of group(groupId)?.options ?? []) {
        const chip = buildChip(option, values.get(option.name), (raw) => set(option.name, raw));
        container.appendChild(chip.el);
        register(option.name, chip.input, { chip: chip.el });
      }
    },
    /** Full labelled fields for the advanced disclosure. */
    renderAdvanced(container, groupId = 'advanced') {
      container.replaceChildren();
      for (const option of group(groupId)?.options ?? []) {
        const field = buildField(option, values.get(option.name), (raw) => set(option.name, raw));
        container.appendChild(field.el);
        register(option.name, field.input, { label: field.label, sync: field.sync });
      }
    },
    /** Current value of one option, e.g. `format`, without assuming it exists. */
    get(name) {
      return values.get(name);
    },
    has(name) {
      return options.has(name);
    },
  };
}

function typeOf(option) {
  const declared = String(option.type ?? '').toLowerCase();
  if (KNOWN_TYPES.has(declared)) return declared;
  // Unknown type from a newer API: fall back to something always editable.
  return Array.isArray(option.values) ? 'enum' : 'string';
}

function normalise(option, raw) {
  const type = typeOf(option);
  if (type === 'boolean') return Boolean(raw);
  if (type === 'integer' || type === 'number') {
    if (raw === '' || raw === null || raw === undefined) return option.default ?? null;
    const n = type === 'integer' ? Math.round(Number(raw)) : Number(raw);
    if (!Number.isFinite(n)) return option.default ?? null;
    return clamp(n, option.min, option.max);
  }
  if (raw === null || raw === undefined) return '';
  return String(raw);
}

function clamp(n, min, max) {
  if (typeof min === 'number' && n < min) return min;
  if (typeof max === 'number' && n > max) return max;
  return n;
}

function sameValue(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < 1e-9;
  return a === b;
}

function writeToInput(input, option, value) {
  if (typeOf(option) === 'boolean') input.checked = Boolean(value);
  else input.value = value === null || value === undefined ? '' : String(value);
}

function stepFor(option) {
  if (typeOf(option) === 'integer') return '1';
  const span = typeof option.max === 'number' ? option.max : 1;
  return span <= 1 ? '0.05' : 'any';
}

function labelFor(option) {
  if (typeof option.label === 'string' && option.label) return option.label;
  return option.name.replaceAll('_', ' ').replace(/^./, (c) => c.toUpperCase());
}

// --------------------------------------------------------------------- chips

function buildChip(option, value, onInput) {
  const el = document.createElement('label');
  el.className = 'chip';
  if (option.help) el.title = `${labelFor(option)} — ${option.help}`;

  const text = document.createElement('span');
  text.className = 'chip__label';
  text.textContent = labelFor(option);
  el.appendChild(text);

  const type = typeOf(option);
  let input;

  if (type === 'enum' || (Array.isArray(option.values) && option.values.length)) {
    input = document.createElement('select');
    input.className = 'chip__control';
    for (const v of option.values ?? []) {
      const opt = document.createElement('option');
      opt.value = String(v);
      opt.textContent = String(v);
      input.appendChild(opt);
    }
    input.value = value === null || value === undefined ? '' : String(value);
    input.addEventListener('change', () => onInput(input.value));
  } else if (type === 'boolean') {
    // A boolean in the basic row still reads best as a two-value select.
    input = document.createElement('select');
    input.className = 'chip__control';
    for (const [v, text2] of [
      ['false', 'off'],
      ['true', 'on'],
    ]) {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = text2;
      input.appendChild(opt);
    }
    input.value = String(Boolean(value));
    input.addEventListener('change', () => onInput(input.value === 'true'));
  } else {
    input = document.createElement('input');
    input.className = 'chip__control';
    if (type === 'integer' || type === 'number') {
      input.type = 'number';
      input.step = stepFor(option);
      if (typeof option.min === 'number') input.min = String(option.min);
      if (typeof option.max === 'number') input.max = String(option.max);
    } else {
      input.type = 'text';
    }
    input.value = value === null || value === undefined ? '' : String(value);
    input.addEventListener('change', () => {
      onInput(input.value);
      writeToInput(input, option, normalise(option, input.value));
    });
  }

  input.setAttribute('aria-label', labelFor(option));
  el.appendChild(input);
  return { el, input };
}

// -------------------------------------------------------------------- fields

function buildField(option, value, onInput) {
  const el = document.createElement('div');
  el.className = 'field';

  const id = `opt-${option.name}`;
  const type = typeOf(option);
  let syncText = null;

  const label = document.createElement('label');
  label.className = 'field__label';
  label.htmlFor = id;
  label.textContent = labelFor(option);
  if (option.help) label.title = option.help;

  let input;

  if (type === 'boolean') {
    const switchEl = document.createElement('label');
    switchEl.className = 'switch';
    input = document.createElement('input');
    input.type = 'checkbox';
    input.id = id;
    input.checked = Boolean(value);
    const track = document.createElement('span');
    track.className = 'switch__track';
    const text = document.createElement('span');
    text.className = 'switch__text';
    text.textContent = input.checked ? 'On' : 'Off';
    input.addEventListener('change', () => {
      text.textContent = input.checked ? 'On' : 'Off';
      onInput(input.checked);
    });
    switchEl.append(input, track, text);
    el.append(label, switchEl);
    syncText = () => {
      text.textContent = input.checked ? 'On' : 'Off';
    };
  } else if (type === 'enum' || (Array.isArray(option.values) && option.values.length)) {
    input = document.createElement('select');
    input.className = 'field__input';
    input.id = id;
    for (const v of option.values ?? []) {
      const opt = document.createElement('option');
      opt.value = String(v);
      opt.textContent = String(v);
      input.appendChild(opt);
    }
    input.value = value === null || value === undefined ? '' : String(value);
    input.addEventListener('change', () => onInput(input.value));
    el.append(label, input);
  } else {
    input = document.createElement('input');
    input.className = 'field__input';
    input.id = id;
    if (type === 'integer' || type === 'number') {
      input.type = 'number';
      input.step = stepFor(option);
      input.inputMode = type === 'integer' ? 'numeric' : 'decimal';
      if (typeof option.min === 'number') input.min = String(option.min);
      if (typeof option.max === 'number') input.max = String(option.max);
    } else {
      input.type = 'text';
    }
    input.value = value === null || value === undefined ? '' : String(value);
    input.addEventListener('change', () => {
      onInput(input.value);
      writeToInput(input, option, normalise(option, input.value));
    });
    el.append(label, input);
  }

  if (option.help) {
    const help = document.createElement('p');
    help.className = 'field__help';
    help.textContent = boundsText(option)
      ? `${option.help} ${boundsText(option)}`
      : option.help;
    el.appendChild(help);
  }

  return { el, input, label, sync: syncText };
}

function boundsText(option) {
  const type = typeOf(option);
  if (type !== 'integer' && type !== 'number') return '';
  const hasMin = typeof option.min === 'number';
  const hasMax = typeof option.max === 'number';
  if (hasMin && hasMax) return `(${option.min}–${option.max})`;
  if (hasMin) return `(min ${option.min})`;
  if (hasMax) return `(max ${option.max})`;
  return '';
}
