<script lang="ts">
  /**
   * Glass text input. Used by the search bar, prompt input, and
   * filename filter. Submits on Enter (when `onSubmit` is set).
   */
  type Props = {
    value: string;
    placeholder?: string;
    name?: string;
    id?: string;
    onInput?: (v: string) => void;
    onSubmit?: (v: string) => void;
    onKeydown?: (e: KeyboardEvent) => void;
  };
  let {
    value = $bindable(''),
    placeholder = '',
    name = 'search',
    id,
    onInput,
    onSubmit,
    onKeydown
  }: Props = $props();

  function handleInput(e: Event) {
    const v = (e.target as HTMLInputElement).value;
    value = v;
    onInput?.(v);
  }
  function handleKeydown(e: KeyboardEvent) {
    onKeydown?.(e);
    if (e.key === 'Enter' && onSubmit) {
      e.preventDefault();
      onSubmit(value);
    }
  }
</script>

<label class="input-wrap">
  <input
    type="text"
    {name}
    {id}
    {placeholder}
    value={value ?? ''}
    oninput={handleInput}
    onkeydown={handleKeydown}
    autocomplete="off"
    spellcheck="false"
    aria-label={placeholder || 'Search'}
  />
</label>

<style>
  .input-wrap {
    display: flex;
    align-items: center;
    gap: var(--s-1);
    background: var(--glass-2);
    border: 1px solid var(--edge-field-color);
    border-radius: var(--r-pill);
    background-image: var(--control-film);
    backdrop-filter: var(--control-blur);
    -webkit-backdrop-filter: var(--control-blur);
    box-shadow: var(--control-texture);
    padding: 0 14px;
    height: 44px;
    backdrop-filter: var(--glass-medium);
    -webkit-backdrop-filter: var(--glass-medium);
    transition: border-color var(--t-fast) var(--ease-out),
                background var(--t-fast) var(--ease-out);
    width: 100%;
  }
  .input-wrap:focus-within {
    border-color: var(--accent);
    background: var(--accent-soft);
  }
  input {
    flex: 1;
    width: 100%;
    height: 100%;
    color: var(--fg-1);
    font-size: var(--fs-md);
  }
  input::placeholder { color: var(--fg-3); }
</style>
