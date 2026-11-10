<script lang="ts">
  /**
   * The error page for the app's routes.
   *
   * It renders INSIDE the (app) layout, so the top bar and the
   * shell are already there — an error on an app route still has
   * the app's navigation around it, exactly as it did before the
   * chrome moved into the route group.
   *
   * The root error page is the one for URLs that match no route at
   * all: it renders at the root, where there is no chrome, so it
   * carries its own.
   */
  import ErrorPage from '$lib/components/ErrorPage.svelte';
</script>

<ErrorPage />
