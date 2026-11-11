#!/usr/bin/env node
/**
 * check-css-tokens.mjs — fail CI if a new hardcoded
 * box-shadow rgba value appears in src/. The token
 * counterparts (--shadow-1, --shadow-2, --shadow-3) live
 * in src/lib/styles/tokens.css.
 *
 * Phase 8: 2 of 3 historical violations were cleaned up
 * (PhotoTile.svelte — exact matches for --shadow-1 /
 * --shadow-3). The 3rd is PromptChips.svelte, which has a
 * single-stop shadow that doesn't match any token and is
 * documented as such inline.
 *
 * If a future commit adds a new rgba(0,0,0,0.0X) box-shadow
 * outside the documented list, this script fails. Add the
 * new site to ALLOWED or replace it with the token.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
// Resolve src/ from CLI arg (--src=path) so tests can run
// the script in a temp dir; default to ../src relative to
// the script (production layout: frontend/scripts/ →
// frontend/src).
const srcArg = process.argv.find((a) => a.startsWith('--src='));
const src = srcArg
  ? path.resolve(srcArg.slice('--src='.length))
  : path.resolve(here, '..', 'src');

// rgba(0,0,0,0.0X) — the four shadow alphas we use as
// tokens (0.04, 0.05, 0.06, 0.08). Adjust this regex if
// the token family grows.
const RGBA = /rgba\(\s*0\s*,\s*0\s*,\s*0\s*,\s*0\.0\d+\s*\)/g;

// Known exceptions: rgba(0,0,0,...) used in places that
// are NOT box-shadow values (e.g. scrim backgrounds,
// overlay fills). Tokens.css is the source of truth and
// must obviously be excluded.
const ALLOWED_PATHS = new Set([
  'lib/styles/tokens.css',
  // theme-dark.css is the DARK theme's token source of truth, the
  // same role tokens.css plays for the light theme. It defines the
  // shadow tokens the dark palette uses, so it is exempt for the
  // same reason tokens.css is.
  'lib/styles/theme-dark.css',
  // PromptChips.svelte has a single-stop box-shadow that
  // doesn't match any --shadow-N token. Documented inline
  // with a comment tying it to --shadow-1's first stop.
  // If a second single-stop site appears, add a new
  // --shadow-0 token instead of expanding this list.
  'lib/components/PromptChips.svelte',
  // global.css .glass uses --glass-inset (rgba 0.7) for
  // the top highlight, which is 0.1 alpha heavier than
  // --shadow-glass's inset (0.6). The two are kept
  // distinct on purpose; the .glass panel sits on
  // different surfaces than the generic --shadow-glass
  // token was tuned for.
  'lib/styles/global.css'
]);

const violations = [];

function walk(dir) {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      walk(full);
    } else if (entry.endsWith('.svelte') || entry.endsWith('.css')) {
      const rel = path.relative(src, full);
      if (ALLOWED_PATHS.has(rel)) continue;
      const text = readFileSync(full, 'utf8');
      // Find rgba matches; check that the containing
      // declaration is box-shadow.
      const lines = text.split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        let m;
        RGBA.lastIndex = 0;
        while ((m = RGBA.exec(line)) !== null) {
          // Look back 3 lines for a `box-shadow:` or
          // `background:` declaration to disambiguate.
          const window = lines.slice(Math.max(0, i - 3), i + 1).join(' ');
          if (/\b(box-shadow|filter|backdrop-filter)\b/.test(window)) {
            violations.push({
              file: rel,
              line: i + 1,
              match: m[0]
            });
          }
        }
      }
    }
  }
}

walk(src);

if (violations.length > 0) {
  console.error('CSS token check failed. Hardcoded box-shadow rgba() in:');
  for (const v of violations) {
    console.error(`  ${v.file}:${v.line}  ${v.match}`);
  }
  console.error('');
  console.error('Use a --shadow-N token (see src/lib/styles/tokens.css).');
  console.error('If this is a genuine single-stop / different shape,');
  console.error('add the path to ALLOWED_PATHS in this script.');
  process.exit(1);
}

console.log('CSS token check: clean.');
