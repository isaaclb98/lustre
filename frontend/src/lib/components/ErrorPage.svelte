<script lang="ts">
  /**
   * The body of an error page — the status, the message and the way
   * back. Shared by the app group's error page and the root one, so
   * an error reads the same wherever it happens.
   *
   * The chrome is NOT part of this. The app group's error page
   * renders inside the (app) layout, which already carries the top
   * bar and the shell; the root one carries the chrome itself,
   * because the root layout has none.
   */
  import { page } from '$app/stores';
  import { resolve } from '$app/paths';

  let { onHome }: { onHome?: () => void } = $props();

  const status = $derived($page.status);
  const message = $derived(
    status === 404
      ? "That page doesn't exist."
      : 'Something went wrong loading this page.'
  );
</script>

<div class="error">
  <h1>{status}</h1>
  <p>{message}</p>
  <a class="home" href={resolve('/')} onclick={onHome}>Back to Lustre</a>
</div>

<style>
  .error {
    /* Centre the block in the space the shell gives it, so the
       status reads as a page rather than as a stray line at the
       top of an empty screen. */
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--s-3);
    text-align: center;
    padding: var(--s-6) var(--s-4);
  }
  h1 {
    font-size: var(--fs-3xl);
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--fg-1);
    margin: 0;
  }
  p {
    margin: 0;
    color: var(--fg-2);
    font-size: var(--fs-md);
  }
  .home {
    margin-top: var(--s-3);
    padding: var(--s-2) var(--s-4);
    border-radius: var(--r-pill);
    border: 1px solid var(--glass-edge);
    background: var(--bg-1);
    color: var(--fg-1);
    text-decoration: none;
    font-size: var(--fs-sm);
    font-weight: 600;
    transition: border-color var(--grid-motion-ms, 100ms) var(--grid-motion-ease, ease),
      background var(--grid-motion-ms, 100ms) var(--grid-motion-ease, ease);
  }
  .home:hover {
    border-color: var(--glass-edge-strong);
    background: var(--bg-2);
  }
</style>
