/**
 * Dev-server hook — stamps the deployment's theme onto the shell.
 *
 * In production the SPA is a STATIC build served by the FastAPI
 * backend, and the backend stamps the theme onto the <html> element
 * of the shell it serves (see search/theme.py). The same marker has
 * to exist in development, or `npm run dev` would always render the
 * light theme and the dark one could only be seen by rebuilding.
 *
 * SvelteKit's dev server renders the shell itself and does NOT run
 * Vite's `transformIndexHtml`, so this hook is the only seam that
 * can patch the served HTML. It is a no-op outside dev: the static
 * build has no server, and the built shell stays theme-neutral so
 * one build can serve either theme.
 *
 * The theme value comes from the environment, which vite.config.ts
 * populates from the REPO-ROOT .env — the same file the backend
 * reads — so one variable drives both processes and the dev server
 * cannot disagree with the backend about which theme is live.
 */
import { dev } from '$app/environment';
import type { Handle } from '@sveltejs/kit';
import { DEFAULT_THEME, THEME_ATTRIBUTE } from '$lib/theme';

/** The themes the frontend ships. Mirrors THEMES in search/theme.py. */
const THEMES = ['light', 'dark'] as const;

/**
 * The configured theme, validated.
 *
 * An unknown value is an error rather than a silent fallback: the
 * backend treats it as a startup error for the same reason — a typo
 * in .env must not look like "the dark theme is broken" when in
 * fact the dark theme was never selected.
 */
function configuredTheme(): string {
  const raw = (process.env.LUSTRE_THEME ?? DEFAULT_THEME).trim();
  if (!THEMES.includes(raw as (typeof THEMES)[number])) {
    throw new Error(
      `LUSTRE_THEME="${raw}" is not a known theme (expected 'light' or 'dark')`
    );
  }
  return raw;
}

export const handle: Handle = async ({ event, resolve }) => {
  const response = await resolve(event);

  // Production has no server; the backend stamps the marker itself.
  if (!dev) return response;

  const contentType = response.headers.get('content-type') ?? '';
  if (!contentType.includes('text/html')) return response;

  const theme = configuredTheme();
  const html = await response.text();
  const stamped = html.replace(
    /<html(\s[^>]*)?>/i,
    (match, attrs: string | undefined) => {
      const without = (attrs ?? '').replace(
        new RegExp(`\\s${THEME_ATTRIBUTE}=("[^"]*"|'[^']*'|[^\\s>]+)`, 'i'),
        ''
      );
      return `<html${without} ${THEME_ATTRIBUTE}="${theme}">`;
    }
  );

  return new Response(stamped, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers
  });
};
