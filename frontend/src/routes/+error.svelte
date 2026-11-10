<script lang="ts">
  /**
   * The root error page — a URL that matches no route at all.
   *
   * It renders at the root, where the layout has no chrome (that
   * is what lets the presentation wall reset to it), so this page
   * carries the app's own navigation itself. A mistyped URL should
   * still land the user somewhere they can navigate from, rather
   * than on a bare page with no way back.
   *
   * The app group's own error page covers errors inside the app's
   * routes; it renders inside the (app) layout and needs no chrome
   * of its own.
   */
  import AppChrome from '$lib/components/AppChrome.svelte';
  import ErrorPage from '$lib/components/ErrorPage.svelte';
</script>

<AppChrome>
  <ErrorPage />
</AppChrome>
