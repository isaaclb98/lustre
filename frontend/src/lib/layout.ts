/**
 * Layout mode — the photo grid's arrangement, chosen at runtime.
 *
 * The frontend is ONE static build shared by every deployment, so the
 * grid's layout cannot be baked into the bundle: a deployment picks
 * between the uniform square grid and Pinterest-style masonry, and
 * switching must not require a rebuild.
 *
 * The backend stamps the deployment's default onto the shell it
 * serves, as a `data-masonry` attribute on <html> (see
 * search/layout.py), and this module reads it back — the same
 * contract the theme uses.
 *
 * This is the DEFAULT, not a lock. The user can override it in the
 * browser (the Settings page), and their choice is remembered there.
 */
import { readable } from 'svelte/store';

/** The attribute the backend stamps the layout mode onto. */
export const MASONRY_ATTRIBUTE = 'data-masonry';

/**
 * Read the masonry mode from an attribute value.
 *
 * Only an explicit "1" or "true" turns masonry on — a missing or
 * unreadable marker means the uniform grid, which is what the app
 * has always rendered.
 */
export function masonryFromAttribute(value: string | null | undefined): boolean {
  return value === '1' || value === 'true';
}

/**
 * The masonry mode the current document was served with.
 *
 * Read from <html data-masonry="...">. Outside the browser there is
 * no marker, so the default applies.
 */
export function readMasonryDefault(): boolean {
  if (typeof document === 'undefined') return false;
  return masonryFromAttribute(
    document.documentElement.getAttribute(MASONRY_ATTRIBUTE)
  );
}

/**
 * The deployment's masonry default, read once at module load.
 *
 * The shell's attribute is in the HTML the browser parsed before any
 * script ran, so it is available here — and it cannot change under
 * us, because a deployment change means a new shell.
 */
export const masonryDefault = readable(readMasonryDefault());

/**
 * The attribute the backend stamps the thumbnail's long edge onto.
 */
export const THUMBNAIL_SIZE_ATTRIBUTE = 'data-thumbnail-size';

/**
 * The thumbnail's long edge, in pixels, for the current deployment.
 *
 * The grid's `srcset` has to state the width the image actually IS,
 * and that is a deployment-level number: the dev stack renders 512,
 * production renders 384. The backend publishes it on the shell (see
 * search/layout.py) and this reads it back.
 *
 * A portrait photo's thumbnail is NARROWER than the long edge — a
 * 512-long-edge portrait is 410 wide — so the long edge is the
 * upper bound on the width, not the width itself. The srcset is
 * therefore built from the bound: a browser that gets a narrower
 * image than advertised still renders it correctly, and one that
 * gets a wider one is never asked to upscale.
 */
export const DEFAULT_THUMBNAIL_SIZE = 384;

/**
 * The attribute the backend stamps the DISPLAY thumbnail's smaller
 * variants onto.
 *
 * These are the dense view's assets: a tile of 180 CSS px needs a
 * far smaller bitmap than a tile of 360, and serving it the
 * canonical wastes both bandwidth and decode time for pixels the
 * browser scales straight back down.
 *
 * The value is a comma-separated list of long edges ("192,384").
 */
export const THUMBNAIL_VARIANTS_ATTRIBUTE = 'data-thumbnail-variants';

/**
 * The deployment's thumbnail variant ladder, read once at module
 * load.
 *
 * Empty when the deployment generates no variants, and then the
 * srcset is a single entry — the canonical — which is exactly what
 * a deployment without variants wants.
 */
export function parseThumbnailVariants(raw: string | null | undefined): number[] {
  if (!raw) return [];
  return raw
    .split(',')
    .map((part) => Number(part.trim()))
    .filter((n) => Number.isFinite(n) && n > 0)
    .sort((a, b) => a - b);
}

export function readThumbnailVariants(): number[] {
  if (typeof document === 'undefined') return [];
  return parseThumbnailVariants(
    document.documentElement.getAttribute(THUMBNAIL_VARIANTS_ATTRIBUTE)
  );
}

export const thumbnailVariants = readable(readThumbnailVariants());

export function readThumbnailSize(): number {
  if (typeof document === 'undefined') return DEFAULT_THUMBNAIL_SIZE;
  const raw = document.documentElement.getAttribute(THUMBNAIL_SIZE_ATTRIBUTE);
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_THUMBNAIL_SIZE;
}

/**
 * The deployment's thumbnail size, read once at module load.
 *
 * The shell's attribute is in the HTML the browser parsed before any
 * script ran, so it is available here.
 */
export const thumbnailSize = readable(readThumbnailSize());

/**
 * The WIDTH of a photo's thumbnail, in pixels.
 *
 * The thumbnail's long edge is `size`, but the thumbnail's width is
 * not: a portrait photo's thumbnail is 384 wide and 512 tall. A
 * `srcset` entry has to state the width the image IS, and stating the
 * long edge instead tells the browser the image is wider than it is
 * — which makes it pick a lower density and render a 135px-wide
 * image into a 240px tile.
 *
 * So the width comes from the photo's own aspect ratio: the thumbnail
 * fits the photo inside a `size` box, so its width is the size
 * itself for a landscape photo and the size times the photo's
 * width/height ratio for a portrait.
 *
 * When the photo's dimensions are unknown, the long edge is the best
 * guess available. Over-stating the width is the safe direction: the
 * browser then treats the image as the smaller asset it is, and
 * never upscales.
 */
export function thumbnailWidthFor(
  size: number,
  sourceWidth?: number | null,
  sourceHeight?: number | null
): number {
  if (!sourceWidth || !sourceHeight || sourceWidth <= 0 || sourceHeight <= 0) {
    return size;
  }
  const ratio = sourceWidth / sourceHeight;
  // A landscape photo's thumbnail is as wide as the long edge; a
  // portrait's is narrower, scaled by the photo's own ratio.
  const width = ratio >= 1 ? size : Math.round(size * ratio);
  return Math.max(1, Math.min(size, width));
}
