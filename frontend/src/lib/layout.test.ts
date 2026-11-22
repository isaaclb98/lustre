/**
 * tests for the layout module — the deployment-level numbers the SPA
 * reads back off the shell, and the geometry derived from them.
 */
import { describe, expect, it } from 'vitest';
import {
  DEFAULT_THUMBNAIL_SIZE,
  LAYOUTS,
  layoutFromAttribute,
  parseThumbnailVariants,
  thumbnailWidthFor
} from './layout';

describe('layoutFromAttribute', () => {
  it('reads the two non-default layouts off an explicit marker', () => {
    expect(layoutFromAttribute('masonry')).toBe('masonry');
    expect(layoutFromAttribute('justified')).toBe('justified');
  });

  it('treats a missing or invalid marker as the square grid', () => {
    // The square grid is the default, and the backend stamps NO
    // marker for it — so a missing marker and an explicit 'grid'
    // have to mean the same thing.
    expect(layoutFromAttribute(null)).toBe('grid');
    expect(layoutFromAttribute(undefined)).toBe('grid');
    expect(layoutFromAttribute('')).toBe('grid');
    expect(layoutFromAttribute('grid')).toBe('grid');
    expect(layoutFromAttribute('0')).toBe('grid');
    expect(layoutFromAttribute('false')).toBe('grid');
    expect(layoutFromAttribute('Masonry')).toBe('grid');
  });
});

describe('LAYOUTS', () => {
  it('names the three arrangements the backend accepts', () => {
    // The frontend's dev-server hook validates the configured mode
    // against this list, and the backend validates its own against
    // LAYOUTS in search/layout.py — so the two have to agree.
    expect(LAYOUTS).toEqual(['grid', 'masonry', 'justified']);
  });
});

describe('parseThumbnailVariants', () => {
  it('is empty when the deployment publishes no ladder', () => {
    // No variants means a single-entry srcset — the canonical — which
    // is what a deployment that generates no variants wants.
    expect(parseThumbnailVariants(null)).toEqual([]);
    expect(parseThumbnailVariants('')).toEqual([]);
  });

  it('reads a comma-separated ladder of long edges', () => {
    expect(parseThumbnailVariants('192,384')).toEqual([192, 384]);
  });

  it('is sorted, so the srcset lists the smallest first', () => {
    expect(parseThumbnailVariants('384,192')).toEqual([192, 384]);
  });

  it('ignores junk rather than breaking the ladder', () => {
    expect(parseThumbnailVariants('192,,bogus,384')).toEqual([192, 384]);
  });
});

describe('thumbnailWidthFor', () => {
  it('is the long edge for a landscape photo', () => {
    // A landscape thumbnail's width IS the long edge.
    expect(thumbnailWidthFor(512, 1200, 800)).toBe(512);
    expect(thumbnailWidthFor(384, 1024, 768)).toBe(384);
  });

  it('is narrower for a portrait photo', () => {
    // A 3:4 portrait's thumbnail is 3/4 as wide as it is tall.
    expect(thumbnailWidthFor(512, 1170, 1560)).toBe(384);
    expect(thumbnailWidthFor(512, 800, 1600)).toBe(256);
  });

  it('is the long edge when the dimensions are unknown', () => {
    // Over-stating the width is the safe direction: the browser then
    // treats the image as the smaller asset it is.
    expect(thumbnailWidthFor(512)).toBe(512);
    expect(thumbnailWidthFor(512, null, null)).toBe(512);
    expect(thumbnailWidthFor(512, 0, 0)).toBe(512);
    expect(thumbnailWidthFor(512, -1, 100)).toBe(512);
  });

  it('never exceeds the long edge', () => {
    for (const [w, h] of [
      [4000, 100],
      [100, 4000],
      [1000, 1000],
      [1170, 1560]
    ] as [number, number][]) {
      expect(thumbnailWidthFor(512, w, h)).toBeLessThanOrEqual(512);
      expect(thumbnailWidthFor(512, w, h)).toBeGreaterThan(0);
    }
  });
});

describe('the default thumbnail size', () => {
  it('matches the backend default', () => {
    // The backend's default (search/config.py) and this one must
    // agree, or a deployment with no THUMBNAIL_SIZE set would build a
    // srcset for a width the backend never renders.
    expect(DEFAULT_THUMBNAIL_SIZE).toBe(640);
  });
});
