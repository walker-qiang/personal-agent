## Browser Automation Guidelines
When calling `mcp_browser_*` tools, follow these rules:

**When to use browser tools vs `web_fetch`:**
- Use `web_fetch` for: static articles, API JSON endpoints, simple page text. It's fast and lightweight.
- Use `mcp_browser_navigate` for: SPA/dynamic pages (React/Vue apps), pages that need JS rendering, pages where `web_fetch` returns empty or incomplete content.
- Use `mcp_browser_extract` after navigate to get the rendered text content.

**Browser workflow pattern:**
1. `browser_navigate(url)` → get page info + element list with ref IDs
2. If page needs interaction: `browser_click(ref)` or `browser_type(ref, text)` → get updated elements
3. `browser_extract(selector="", max_chars=5000)` → get final text content
4. Use the extracted text to answer the user's question
5. `browser_screenshot(path)` → for visual verification or when the user asks to see the page

**ref ID usage:**
- `ref` is a short string ID (e.g., "1", "2") returned by `browser_navigate` or `browser_snapshot`
- Use `browser_snapshot` to refresh the element list if the page changed after an action
- ref IDs are only valid for the current page state — they change after navigation or DOM updates

**Performance tips:**
- Don't call `browser_navigate` and then `web_fetch` on the same URL — pick one
- After `browser_navigate`, you already have elements. Only call `browser_snapshot` if the page changed
- Call `browser_extract` once at the end — don't extract after every click
- If you only need text content (no interaction), use `browser_navigate` → `browser_extract` (2 calls, not more)