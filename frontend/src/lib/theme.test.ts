/**
 * Unit tests for the theme contract — the module every other
 * theme-aware piece reads from.
 *
 * The contract is deliberately tiny: a `data-theme` attribute on
 * <html>, holding 'light' or 'dark'. The backend stamps it onto the
 * shell it serves (search/theme.py); the CSS selects the palette off
 * it; the grid geometry reads it in JS.
 *
 * What we pin:
 *   - the attribute name and the two theme names,
 *   - an unreadable or unknown marker resolves to the default rather
 *     than to undefined (a bad marker must not produce a broken
 *     theme),
 *   - the store tracks the attribute when it changes, so a test (or
 *     a browser console) can switch themes at runtime.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  DEFAULT_THEME,
  THEME_ATTRIBUTE,
  currentTheme,
  readTheme,
  theme,
  themeFromAttribute
} from './theme';

describe('theme contract', () => {
  afterEach(() => {
    document.documentElement.removeAttribute(THEME_ATTRIBUTE);
  });

  describe('themeFromAttribute', () => {
    it('reads the two known themes', () => {
      expect(themeFromAttribute('light')).toBe('light');
      expect(themeFromAttribute('dark')).toBe('dark');
    });

    it('falls back to the default for anything unreadable', () => {
      // A missing marker, an empty marker, an unknown name, and a
      // case difference all resolve to the default. The marker is
      // written by the backend, which validates its own value, so an
      // unknown name here means something else went wrong — and the
      // safe reading of that is the default theme, not undefined.
      expect(themeFromAttribute(null)).toBe(DEFAULT_THEME);
      expect(themeFromAttribute(undefined)).toBe(DEFAULT_THEME);
      expect(themeFromAttribute('')).toBe(DEFAULT_THEME);
      expect(themeFromAttribute('sepia')).toBe(DEFAULT_THEME);
      expect(themeFromAttribute('DARK')).toBe(DEFAULT_THEME);
    });
  });

  describe('readTheme', () => {
    it('reads the marker off <html>', () => {
      document.documentElement.setAttribute(THEME_ATTRIBUTE, 'dark');
      expect(readTheme()).toBe('dark');
    });

    it('reads the default when no marker is present', () => {
      document.documentElement.removeAttribute(THEME_ATTRIBUTE);
      expect(readTheme()).toBe(DEFAULT_THEME);
    });
  });

  describe('currentTheme', () => {
    it('reads the same marker the store watches', () => {
      document.documentElement.setAttribute(THEME_ATTRIBUTE, 'dark');
      expect(currentTheme()).toBe('dark');
      document.documentElement.setAttribute(THEME_ATTRIBUTE, 'light');
      expect(currentTheme()).toBe('light');
    });
  });

  describe('the theme store', () => {
    let seen: string[] = [];

    beforeEach(() => {
      seen = [];
      document.documentElement.removeAttribute(THEME_ATTRIBUTE);
    });

    it('reports the marker as it changes', async () => {
      const unsubscribe = theme.subscribe((value) => seen.push(value));

      // The store's first value is the marker at subscribe time.
      expect(seen).toEqual([DEFAULT_THEME]);

      document.documentElement.setAttribute(THEME_ATTRIBUTE, 'dark');
      // The MutationObserver delivers asynchronously.
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(seen).toContain('dark');

      document.documentElement.setAttribute(THEME_ATTRIBUTE, 'light');
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(seen).toContain('light');

      unsubscribe();
    });
  });
});
