"""浏览器自动化 MCP 服务器：基于 Playwright 的页面导航、交互、提取。

启动方式：python browser_tools.py
依赖：playwright (pip install playwright && playwright install chromium)

环境变量：
  BROWSER_HEADLESS         - 是否无头模式 (默认 true)
  BROWSER_DEFAULT_TIMEOUT  - 默认超时毫秒 (默认 30000)
  BROWSER_MAX_OUTPUT_CHARS - 文本输出最大字符数 (默认 20000)
  BROWSER_USER_AGENT       - 自定义 User-Agent
  BROWSER_VIEWPORT_WIDTH   - 视口宽度 (默认 1280)
  BROWSER_VIEWPORT_HEIGHT  - 视口高度 (默认 720)
  BROWSER_BLOCK_RESOURCES  - 拦截图片/CSS/字体等资源以加速加载 (默认 true)
  BROWSER_MAX_ELEMENTS     - 快照返回最大元素数 (默认 50)
"""

from __future__ import annotations

import base64
import logging
import os
import time as _time
from typing import Any

logger = logging.getLogger("browser-mcp")

# ---- Playwright lazy import ----
try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

    _PW_AVAILABLE = True
except ImportError:
    _PW_AVAILABLE = False
    logger.warning("playwright not installed; browser tools disabled")

# ---- Config from env ----
_HEADLESS = os.environ.get("BROWSER_HEADLESS", "true").strip().lower() in ("1", "true", "yes")
_DEFAULT_TIMEOUT = int(os.environ.get("BROWSER_DEFAULT_TIMEOUT", "30000"))
_MAX_OUTPUT_CHARS = int(os.environ.get("BROWSER_MAX_OUTPUT_CHARS", "20000"))
_USER_AGENT = os.environ.get(
    "BROWSER_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)
_VIEWPORT_WIDTH = int(os.environ.get("BROWSER_VIEWPORT_WIDTH", "1280"))
_VIEWPORT_HEIGHT = int(os.environ.get("BROWSER_VIEWPORT_HEIGHT", "720"))
# Block heavy resources (images, CSS, fonts) for faster page loads when only extracting text
_BLOCK_RESOURCES = os.environ.get("BROWSER_BLOCK_RESOURCES", "true").strip().lower() in ("1", "true", "yes")
# Max interactive elements to return in snapshot (prevents LLM context overflow on complex pages)
_MAX_ELEMENTS = int(os.environ.get("BROWSER_MAX_ELEMENTS", "50"))

# ---- URL safety ----
_BLOCKED_PROTOCOLS = {"file:", "javascript:", "data:", "chrome:", "about:"}
_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254", "[::1]"}

# ---- Global state (module-level, single process) ----
_pw: Playwright | None = None
_browser: Browser | None = None
_page: Page | None = None
# ref_id -> element info dict (tag, text, role, selector)
# Selector uses [data-mcp-ref="N"] which we inject into the page
_last_snapshot_elements: list[dict[str, Any]] = []
# Idle timeout: close browser after 5 minutes of inactivity
_IDLE_TIMEOUT_SEC = 300
_last_activity: float = 0.0


def _validate_url(url: str) -> tuple[bool, str]:
    """校验 URL 安全性。"""
    if not url or not url.strip():
        return (False, "URL 为空")
    url_lower = url.lower().strip()
    for proto in _BLOCKED_PROTOCOLS:
        if url_lower.startswith(proto):
            return (False, f"协议被禁止: {proto}")
    for host in _BLOCKED_HOSTS:
        if host in url_lower:
            return (False, f"目标地址被禁止: {host}")
    if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
        return (False, "URL 必须以 http:// 或 https:// 开头")
    return (True, "")


async def _ensure_browser() -> Page:
    """确保浏览器实例已启动，返回 Page 对象。"""
    global _pw, _browser, _page, _last_activity

    # Idle timeout: close browser if idle for too long
    if _browser and _last_activity > 0:
        idle_sec = _time.monotonic() - _last_activity
        if idle_sec > _IDLE_TIMEOUT_SEC:
            logger.info("browser: idle for %.0fs, closing to free resources", idle_sec)
            try:
                if _page and not _page.is_closed():
                    await _page.close()
                if _browser:
                    for ctx in _browser.contexts:
                        await ctx.close()
            except Exception:
                pass
            _browser = None
            _page = None

    if _page and not _page.is_closed():
        _last_activity = _time.monotonic()
        return _page

    # Restart if page was closed or browser crashed
    if _browser and not _browser.is_connected():
        try:
            await _browser.close()
        except Exception:
            pass
        _browser = None
        _page = None

    if not _pw:
        _pw = await async_playwright().start()

    if not _browser:
        _browser = await _pw.chromium.launch(headless=_HEADLESS)

    context = await _browser.new_context(
        user_agent=_USER_AGENT,
        viewport={"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT},
        locale="zh-CN",
    )
    context.set_default_timeout(_DEFAULT_TIMEOUT)

    # Performance: block heavy resources when only extracting text content
    if _BLOCK_RESOURCES:
        async def _block_routes(route):
            if route.request.resource_type in ("image", "media", "font", "stylesheet"):
                await route.abort()
            else:
                await route.continue_()
        await context.route("**/*", _block_routes)

    _page = await context.new_page()
    _last_activity = _time.monotonic()
    logger.info("browser: new page created (headless=%s)", _HEADLESS)
    return _page


async def _inject_refs(page: Page) -> list[dict[str, Any]]:
    """注入 data-mcp-ref 属性到页面可交互元素，返回元素信息列表。"""
    js_code = """
    (maxElements) => {
        // 查找所有可交互元素
        const selector = [
            'a[href]', 'button', 'input', 'textarea', 'select',
            '[role="button"]', '[role="link"]', '[role="tab"]',
            '[role="menuitem"]', '[role="checkbox"]', '[role="radio"]',
            '[onclick]', '[contenteditable]',
            'summary', 'details',
        ].join(', ');

        const elements = Array.from(document.querySelectorAll(selector));
        const results = [];

        // 清除旧的 ref 标记
        document.querySelectorAll('[data-mcp-ref]').forEach(el => {
            el.removeAttribute('data-mcp-ref');
        });

        let refId = 0;
        for (const el of elements) {
            if (refId >= maxElements) break;

            // 跳过不可见元素
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 && rect.height === 0) continue;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') continue;

            refId++;
            el.setAttribute('data-mcp-ref', String(refId));

            // 提取元素信息
            const tag = el.tagName.toLowerCase();
            const text = (el.innerText || el.textContent || '').trim().substring(0, 100);
            const role = el.getAttribute('role') || '';
            const type = el.getAttribute('type') || '';
            const placeholder = el.getAttribute('placeholder') || '';
            const href = el.getAttribute('href') || '';
            const ariaLabel = el.getAttribute('aria-label') || '';

            results.push({
                ref: String(refId),
                tag: tag,
                text: text,
                role: role,
                type: type,
                placeholder: placeholder,
                href: href,
                aria_label: ariaLabel,
            });
        }
        return results;
    }
    """
    elements = await page.evaluate(js_code, _MAX_ELEMENTS)
    return elements if isinstance(elements, list) else []


async def _get_page_info(page: Page) -> dict[str, Any]:
    """获取当前页面的基本信息。"""
    title = await page.title()
    url = page.url
    return {"url": url, "title": title}


async def _refresh_snapshot() -> list[dict[str, Any]]:
    """刷新页面快照并存储到全局变量。"""
    global _last_snapshot_elements
    if not _page or _page.is_closed():
        _last_snapshot_elements = []
        return []
    _last_snapshot_elements = await _inject_refs(_page)
    return _last_snapshot_elements


def _truncate(text: str, max_chars: int | None = None) -> str:
    """截断文本到最大长度。"""
    limit = max_chars or _MAX_OUTPUT_CHARS
    if len(text) > limit:
        return text[:limit] + "\n\n... (内容已截断)"
    return text


# =====================================================================
# MCP Tool Definitions
# =====================================================================

if _PW_AVAILABLE:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("browser-tools")

    @mcp.tool()
    async def browser_navigate(url: str, wait_until: str = "domcontentloaded") -> dict:
        """导航到指定 URL 并返回页面快照。

        适用于：动态渲染页面、SPA 应用、需要交互的页面。
        比 web_fetch 慢但功能更强大（支持 JS 渲染、点击、填表）。

        Args:
            url: 目标页面 URL（必须是 http/https）
            wait_until: 等待策略 (domcontentloaded | networkidle | load)

        Returns:
            包含 url, title, elements 列表的字典
        """
        ok, msg = _validate_url(url)
        if not ok:
            return {"error": True, "message": msg}

        valid_waits = {"domcontentloaded", "networkidle", "load"}
        if wait_until not in valid_waits:
            wait_until = "domcontentloaded"

        try:
            page = await _ensure_browser()
            response = await page.goto(url, wait_until=wait_until, timeout=_DEFAULT_TIMEOUT)
            # 等待页面稳定
            await page.wait_for_load_state("domcontentloaded", timeout=10000)

            elements = await _refresh_snapshot()
            info = await _get_page_info(page)
            status = response.status if response else 0

            return {
                "url": info["url"],
                "title": info["title"],
                "status": status,
                "elements": elements,
                "element_count": len(elements),
            }
        except Exception as exc:
            logger.error("browser_navigate failed: %s", exc)
            return {"error": True, "message": f"导航失败: {exc}"}

    @mcp.tool()
    async def browser_snapshot() -> dict:
        """获取当前页面的可交互元素列表。

        在 navigate 之后页面发生变化时（如 AJAX 加载、动画完成）使用此工具刷新引用。

        Returns:
            包含 url, title, elements 列表的字典
        """
        try:
            page = await _ensure_browser()
            elements = await _refresh_snapshot()
            info = await _get_page_info(page)
            return {
                "url": info["url"],
                "title": info["title"],
                "elements": elements,
                "element_count": len(elements),
            }
        except Exception as exc:
            logger.error("browser_snapshot failed: %s", exc)
            return {"error": True, "message": f"获取快照失败: {exc}"}

    @mcp.tool()
    async def browser_click(ref: str) -> dict:
        """点击页面上指定元素。

        Args:
            ref: 元素引用 ID（来自 navigate 或 snapshot 返回的 elements 列表）

        Returns:
            包含 success, url_after, title_after, elements 列表的字典
        """
        if not ref or not ref.strip():
            return {"error": True, "message": "ref 不能为空"}

        try:
            page = await _ensure_browser()
            selector = f'[data-mcp-ref="{ref}"]'

            # 检查元素是否存在
            element = await page.query_selector(selector)
            if not element:
                # 元素可能已失效，刷新快照后重试
                await _refresh_snapshot()
                element = await page.query_selector(selector)
                if not element:
                    return {
                        "error": True,
                        "message": f"未找到 ref={ref} 的元素。请先调用 browser_snapshot 获取最新元素列表。",
                    }

            await element.click(timeout=_DEFAULT_TIMEOUT)
            # 等待页面可能的导航或更新
            await page.wait_for_load_state("domcontentloaded", timeout=10000)

            elements = await _refresh_snapshot()
            info = await _get_page_info(page)
            return {
                "success": True,
                "url_after": info["url"],
                "title_after": info["title"],
                "elements": elements,
                "element_count": len(elements),
            }
        except Exception as exc:
            logger.error("browser_click failed (ref=%s): %s", ref, exc)
            return {"error": True, "message": f"点击失败: {exc}"}

    @mcp.tool()
    async def browser_type(ref: str, text: str, submit: bool = False) -> dict:
        """在指定输入元素中输入文本。

        Args:
            ref: 输入框元素引用 ID（来自 navigate 或 snapshot）
            text: 要输入的文本
            submit: 是否在输入后按回车提交表单（默认 False）

        Returns:
            包含 success, url_after, title_after, elements 列表的字典
        """
        if not ref or not ref.strip():
            return {"error": True, "message": "ref 不能为空"}
        if text is None:
            return {"error": True, "message": "text 不能为空"}

        try:
            page = await _ensure_browser()
            selector = f'[data-mcp-ref="{ref}"]'

            element = await page.query_selector(selector)
            if not element:
                await _refresh_snapshot()
                element = await page.query_selector(selector)
                if not element:
                    return {
                        "error": True,
                        "message": f"未找到 ref={ref} 的元素。请先调用 browser_snapshot 获取最新元素列表。",
                    }

            # 清空现有内容
            await element.click()
            await element.fill("")

            # 输入新文本
            await element.type(text, delay=50)

            if submit:
                await page.keyboard.press("Enter")
                await page.wait_for_load_state("domcontentloaded", timeout=10000)

            elements = await _refresh_snapshot()
            info = await _get_page_info(page)
            return {
                "success": True,
                "url_after": info["url"],
                "title_after": info["title"],
                "elements": elements,
                "element_count": len(elements),
            }
        except Exception as exc:
            logger.error("browser_type failed (ref=%s): %s", ref, exc)
            return {"error": True, "message": f"输入失败: {exc}"}

    @mcp.tool()
    async def browser_extract(selector: str = "", max_chars: int = 5000) -> dict:
        """提取当前页面的文本内容。

        适用于：在导航和交互后提取最终页面内容。
        比 web_fetch 更强大，能提取 JS 渲染后的动态内容。

        Args:
            selector: CSS 选择器（为空则提取整个页面的可见文本）
            max_chars: 最大返回字符数（默认 5000，上限 20000）

        Returns:
            包含 url, text, length 的字典
        """
        max_chars = min(max(max_chars, 500), _MAX_OUTPUT_CHARS)

        try:
            page = await _ensure_browser()
            info = await _get_page_info(page)

            if selector:
                # 按 CSS 选择器提取
                element = await page.query_selector(selector)
                if not element:
                    return {
                        "url": info["url"],
                        "error": True,
                        "message": f"未找到匹配选择器的元素: {selector}",
                    }
                text = await element.inner_text()
            else:
                # 提取整个页面的可见文本
                text = await page.evaluate("""
                    () => {
                        // 移除 script, style, noscript 标签内容
                        const clone = document.body.cloneNode(true);
                        clone.querySelectorAll('script, style, noscript, svg').forEach(el => el.remove());
                        return clone.innerText || clone.textContent || '';
                    }
                """)

            text = text.strip() if text else ""
            text = _truncate(text, max_chars)

            return {
                "url": info["url"],
                "text": text,
                "length": len(text),
            }
        except Exception as exc:
            logger.error("browser_extract failed: %s", exc)
            return {"error": True, "message": f"提取失败: {exc}"}

    @mcp.tool()
    async def browser_screenshot(full_page: bool = False) -> dict:
        """截取当前页面的截图。

        适用于：调试页面状态、验证操作结果、记录页面快照。

        Args:
            full_page: 是否截取完整页面（默认只截视口区域）

        Returns:
            包含 url, screenshot_base64, width, height 的字典
        """
        try:
            page = await _ensure_browser()
            info = await _get_page_info(page)

            screenshot_bytes = await page.screenshot(
                full_page=full_page,
                type="png",
            )

            # 限制截图大小（base64 编码后约 1.3x 原始大小）
            max_bytes = 1_000_000  # ~1MB
            if len(screenshot_bytes) > max_bytes:
                # 重新截取更小的图片
                screenshot_bytes = await page.screenshot(
                    full_page=False,
                    type="jpeg",
                    quality=70,
                )

            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            viewport = page.viewport_size or {"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT}

            return {
                "url": info["url"],
                "screenshot_base64": screenshot_b64,
                "width": viewport["width"],
                "height": viewport["height"],
                "format": "png" if len(screenshot_bytes) <= max_bytes else "jpeg",
            }
        except Exception as exc:
            logger.error("browser_screenshot failed: %s", exc)
            return {"error": True, "message": f"截图失败: {exc}"}

    # =================================================================
    # Phase 2: Extended tools
    # =================================================================

    @mcp.tool()
    async def browser_wait_for(text: str = "", selector: str = "", timeout: int = 10000) -> dict:
        """等待页面上出现特定文本或元素。

        适用于：等待 AJAX 加载完成、等待动态内容出现。

        Args:
            text: 等待出现的文本内容（与 selector 二选一）
            selector: 等待出现的 CSS 选择器（与 text 二选一）
            timeout: 超时时间（毫秒，默认 10000）

        Returns:
            包含 success, url, title 的字典
        """
        if not text and not selector:
            return {"error": True, "message": "必须提供 text 或 selector 参数"}

        try:
            page = await _ensure_browser()
            timeout = min(max(timeout, 1000), 30000)

            if selector:
                await page.wait_for_selector(selector, timeout=timeout, state="visible")
            elif text:
                # 使用 Playwright 的 text 定位器
                locator = page.get_by_text(text, exact=False)
                await locator.wait_for(timeout=timeout, state="visible")

            info = await _get_page_info(page)
            return {
                "success": True,
                "url": info["url"],
                "title": info["title"],
            }
        except Exception as exc:
            logger.error("browser_wait_for failed: %s", exc)
            return {"error": True, "message": f"等待超时: {exc}"}

    @mcp.tool()
    async def browser_press_key(key: str) -> dict:
        """模拟按键操作。

        适用于：Enter 提交、Tab 切换焦点、Escape 关闭弹窗等。

        Args:
            key: 按键名称（如 Enter, Tab, Escape, ArrowDown, Backspace）

        Returns:
            包含 success, url_after, title_after 的字典
        """
        if not key or not key.strip():
            return {"error": True, "message": "key 不能为空"}

        try:
            page = await _ensure_browser()
            await page.keyboard.press(key.strip())
            await page.wait_for_load_state("domcontentloaded", timeout=5000)

            info = await _get_page_info(page)
            return {
                "success": True,
                "url_after": info["url"],
                "title_after": info["title"],
            }
        except Exception as exc:
            logger.error("browser_press_key failed (key=%s): %s", key, exc)
            return {"error": True, "message": f"按键失败: {exc}"}

    @mcp.tool()
    async def browser_scroll(direction: str = "down", amount: int = 500) -> dict:
        """滚动页面。

        适用于：触发懒加载内容、查看长页面下方内容。

        Args:
            direction: 滚动方向 (up | down)
            amount: 滚动像素数（默认 500）

        Returns:
            包含 success, scroll_y 的字典
        """
        direction = direction.lower().strip()
        if direction not in ("up", "down"):
            return {"error": True, "message": "direction 必须是 up 或 down"}

        try:
            page = await _ensure_browser()
            amount = min(max(amount, 100), 5000)
            sign = 1 if direction == "down" else -1

            scroll_y = await page.evaluate(f"""
                () => {{
                    window.scrollBy(0, {sign * amount});
                    return window.scrollY;
                }}
            """)

            # 短暂等待可能的懒加载
            await page.wait_for_timeout(500)

            return {
                "success": True,
                "scroll_y": scroll_y,
            }
        except Exception as exc:
            logger.error("browser_scroll failed: %s", exc)
            return {"error": True, "message": f"滚动失败: {exc}"}

    @mcp.tool()
    async def browser_select_option(ref: str, value: str) -> dict:
        """选择下拉框选项。

        Args:
            ref: select 元素引用 ID
            value: 要选择的选项值

        Returns:
            包含 success, selected_value 的字典
        """
        if not ref or not value:
            return {"error": True, "message": "ref 和 value 不能为空"}

        try:
            page = await _ensure_browser()
            selector = f'[data-mcp-ref="{ref}"]'

            element = await page.query_selector(selector)
            if not element:
                await _refresh_snapshot()
                element = await page.query_selector(selector)
                if not element:
                    return {
                        "error": True,
                        "message": f"未找到 ref={ref} 的元素。请先调用 browser_snapshot。",
                    }

            # 使用 Playwright 的 select_option
            await element.select_option(value)
            await page.wait_for_load_state("domcontentloaded", timeout=5000)

            selected = await element.evaluate("el => el.value")

            return {
                "success": True,
                "selected_value": selected,
            }
        except Exception as exc:
            logger.error("browser_select_option failed (ref=%s): %s", ref, exc)
            return {"error": True, "message": f"选择失败: {exc}"}

    @mcp.tool()
    async def browser_get_cookies() -> dict:
        """获取当前页面的 Cookie 列表（用于调试和会话管理）。

        Returns:
            包含 cookies 列表的字典
        """
        try:
            page = await _ensure_browser()
            context = page.context
            cookies = await context.cookies()
            # 只返回关键信息，避免泄露过多细节
            safe_cookies = [
                {
                    "name": c.get("name", ""),
                    "domain": c.get("domain", ""),
                    "path": c.get("path", ""),
                    "expires": c.get("expires", -1),
                    "secure": c.get("secure", False),
                    "http_only": c.get("httpOnly", False),
                    "same_site": c.get("sameSite", ""),
                }
                for c in cookies
            ]
            return {
                "url": page.url,
                "cookie_count": len(safe_cookies),
                "cookies": safe_cookies,
            }
        except Exception as exc:
            logger.error("browser_get_cookies failed: %s", exc)
            return {"error": True, "message": f"获取 Cookie 失败: {exc}"}

    @mcp.tool()
    async def browser_save_state(storage_path: str = "") -> dict:
        """保存当前浏览器状态（Cookie + localStorage）到文件。

        适用于：保存登录态，下次启动时恢复。

        Args:
            storage_path: 保存路径（默认保存到 var/mcp/browser_state.json）

        Returns:
            包含 success, saved_path 的字典
        """
        if not storage_path:
            storage_path = os.path.join(os.path.dirname(__file__), "browser_state.json")

        try:
            page = await _ensure_browser()
            context = page.context
            await context.storage_state(path=storage_path)
            return {
                "success": True,
                "saved_path": storage_path,
                "message": f"浏览器状态已保存到 {storage_path}",
            }
        except Exception as exc:
            logger.error("browser_save_state failed: %s", exc)
            return {"error": True, "message": f"保存状态失败: {exc}"}

    @mcp.tool()
    async def browser_restore_state(storage_path: str = "") -> dict:
        """从文件恢复浏览器状态（Cookie + localStorage）。

        需要在 navigate 之前调用。恢复后再次导航可保持登录态。

        Args:
            storage_path: 状态文件路径（默认 var/mcp/browser_state.json）

        Returns:
            包含 success 的字典
        """
        if not storage_path:
            storage_path = os.path.join(os.path.dirname(__file__), "browser_state.json")

        if not os.path.isfile(storage_path):
            return {"error": True, "message": f"状态文件不存在: {storage_path}"}

        try:
            global _browser, _page
            page = await _ensure_browser()

            # 需要创建新的 context 来加载状态
            old_context = page.context
            new_context = await _browser.new_context(
                user_agent=_USER_AGENT,
                viewport={"width": _VIEWPORT_WIDTH, "height": _VIEWPORT_HEIGHT},
                locale="zh-CN",
                storage_state=storage_path,
            )
            new_context.set_default_timeout(_DEFAULT_TIMEOUT)

            # 关闭旧 context，切换到新的
            await old_context.close()
            _page = await new_context.new_page()

            return {
                "success": True,
                "message": f"已从 {storage_path} 恢复浏览器状态",
            }
        except Exception as exc:
            logger.error("browser_restore_state failed: %s", exc)
            return {"error": True, "message": f"恢复状态失败: {exc}"}

    @mcp.tool()
    async def browser_close() -> dict:
        """关闭当前浏览器实例，释放资源。

        适用于：完成所有操作后清理资源，或浏览器异常时重置。
        """
        global _browser, _page, _last_activity
        try:
            if _page and not _page.is_closed():
                await _page.close()
            if _browser:
                context_pages = _browser.contexts
                for ctx in context_pages:
                    for p in ctx.pages:
                        await p.close()
                    await ctx.close()
            _page = None
            _browser = None
            _last_activity = 0.0
            return {"success": True, "message": "浏览器已关闭"}
        except Exception as exc:
            logger.error("browser_close failed: %s", exc)
            _page = None
            _browser = None
            _last_activity = 0.0
            return {"error": True, "message": f"关闭失败: {exc}"}


if __name__ == "__main__":
    if not _PW_AVAILABLE:
        print("ERROR: playwright is not installed.")
        print("Install with: pip install playwright && playwright install chromium")
        raise SystemExit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    logger.info("browser MCP server starting (headless=%s, timeout=%dms)", _HEADLESS, _DEFAULT_TIMEOUT)
    mcp.run()
