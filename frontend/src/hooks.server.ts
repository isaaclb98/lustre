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
import { createHash } from 'node:crypto';
import { dev } from '$app/environment';
import type { Handle } from '@sveltejs/kit';
import { DEFAULT_THEME, THEME_ATTRIBUTE } from '$lib/theme';
import {
  DEFAULT_THUMBNAIL_SIZE,
  LAYOUTS,
  LAYOUT_ATTRIBUTE,
  THUMBNAIL_SIZE_ATTRIBUTE
} from '$lib/layout';

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

/**
 * The configured grid layout.
 *
 * The same value the backend stamps onto the shell it serves, read
 * from the same environment — LUSTRE_LAYOUT, one of 'grid',
 * 'masonry' or 'justified'.
 *
 * An unknown value is an error rather than a silent fallback, for
 * the same reason the theme is: a typo in .env must not look like
 * "the dark theme is broken" when in fact the dark theme was never
 * selected.
 *
 * The legacy LUSTRE_MASONRY boolean is still honoured, and only
 * when LUSTRE_LAYOUT is absent — a deployment that set it to 1 gets
 * the masonry layout, so an existing .env keeps working.
 */
function configuredLayout(): string {
  const raw = (process.env.LUSTRE_LAYOUT ?? '').trim().toLowerCase();
  if (!raw) {
    const legacy = (process.env.LUSTRE_MASONRY ?? '').trim().toLowerCase();
    return legacy === '1' || legacy === 'true' || legacy === 'yes'
      ? 'masonry'
      : 'grid';
  }
  if (!LAYOUTS.includes(raw as (typeof LAYOUTS)[number])) {
    throw new Error(
      `LUSTRE_LAYOUT="${raw}" is not a known layout (expected 'grid', 'masonry' or 'justified')`
    );
  }
  return raw;
}

/**
 * The configured thumbnail size.
 *
 * The same value the backend stamps onto the shell it serves, read
 * from the same environment. The grid's srcset has to state the
 * width the image IS, and that is the deployment's thumbnail size.
 */
function configuredThumbnailSize(): number {
  const raw = (process.env.THUMBNAIL_SIZE ?? '').trim();
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_THUMBNAIL_SIZE;
}

export const handle: Handle = async ({ event, resolve }) => {
  const response = await resolve(event);

  // Production has no server; the backend stamps the marker itself.
  if (!dev) return response;

  const contentType = response.headers.get('content-type') ?? '';
  if (!contentType.includes('text/html')) return response;

  const theme = configuredTheme();
  const layout = configuredLayout();
  const thumbnailSize = configuredThumbnailSize();
  const html = await response.text();
  const stamped = html.replace(
    /<html(\s[^>]*)?>/i,
    (match, attrs: string | undefined) => {
      const without = (attrs ?? '')
        .replace(
          new RegExp(`\\s${THEME_ATTRIBUTE}=("[^"]*"|'[^']*'|[^\\s>]+)`, 'i'),
          ''
        )
        .replace(
          new RegExp(`\\s${LAYOUT_ATTRIBUTE}=("[^"]*"|'[^']*'|[^\\s>]+)`, 'i'),
          ''
        )
        .replace(
          new RegExp(
            `\\s${THUMBNAIL_SIZE_ATTRIBUTE}=("[^"]*"|'[^']*'|[^\\s>]+)`,
            'i'
          ),
          ''
        );
      // The square grid is the default, and the default carries no
      // marker — the same rule the backend's inject_layout follows.
      const marker =
        layout === 'grid' ? '' : ` ${LAYOUT_ATTRIBUTE}="${layout}"`;
      return `<html${without} ${THEME_ATTRIBUTE}="${theme}"${marker} ${THUMBNAIL_SIZE_ATTRIBUTE}="${thumbnailSize}">`;
    }
  );

  // The stamped body is LONGER than the original — the markers are
  // added to it — so the original's content-length would cut the
  // response off mid-document, and the original's ETag would let a
  // browser revalidate against bytes that no longer exist (a 304
  // serves the STALE body, which is the one that was truncated).
  // Recompute both.
  const headers = new Headers(response.headers);
  headers.delete('content-length');
  headers.set('etag', `"${createHash('sha1').update(stamped).digest('hex')}"`);

  return new Response(stamped, {
    status: response.status,
    statusText: response.statusText,
    headers
  });
};
