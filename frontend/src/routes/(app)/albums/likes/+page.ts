/**
 * The built-in album used to live at /albums/likes. It is now
 * called Favourites and lives at /albums/favourites.
 *
 * Old links and bookmarks should land on the renamed album rather
 * than on a "not found" page, so this route redirects instead of
 * rendering anything of its own.
 */
import { redirect } from '@sveltejs/kit';
import type { PageLoad } from './$types';

export const load: PageLoad = async () => {
  redirect(308, '/albums/favourites');
};
