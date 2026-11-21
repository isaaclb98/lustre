/**
 * tests for the layout module — the deployment-level numbers the SPA
 * reads back off the shell, and the geometry derived from them.
 */
import { describe, expect, it } from 'vitest';
import {
  DEFAULT_THUMBNAIL_SIZE,
  masonryFromAttribute,
  thumbnailWidthFor
} from './layout';

describe('masonryFromAttribute', () => {
  it('turns masonry on only for an explicit marker', () => {
    expect(masonryFromAttribute('1')).toBe(true);
    expect(masonryFromAttribute('true')).toBe(true);
  });

  it('treats a missing or invalid marker as the square grid', () => {
    expect(masonryFromAttribute(null)).toBe(false);
    expect(masonryFromAttribute(undefined)).toBe(false);
    expect(masonryFromAttribute('')).toBe(false);
    expect(masonryFromAttribute('0')).toBe(false);
    expect(masonryFromAttribute('false')).toBe(false);
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
    expect(DEFAULT_THUMBNAIL_SIZE).toBe(384);
  });
});
