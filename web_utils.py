import asyncio
import logging
import os
import re
import json
from typing import List, Optional, Dict, Any, Callable, Awaitable, Tuple, Set
from bs4 import BeautifulSoup
import random
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib

import aiohttp
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError # type: ignore
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound # type: ignore
import xml.etree.ElementTree

# Assuming config is imported from config.py
from config import config
from utils import cleanup_playwright_processes
from common_models import TweetData, GroundNewsArticle

logger = logging.getLogger(__name__)

PLAYWRIGHT_SEM = asyncio.Semaphore(config.PLAYWRIGHT_MAX_CONCURRENCY)

async def _graceful_close_playwright(page: Optional[Any], context: Optional[Any], browser: Optional[Any], profile_dir_usable: bool, timeout: float = 1.0) -> None:
    """Attempt to close Playwright objects; kill lingering processes if they remain."""
    if page and not getattr(page, "is_closed", lambda: True)():
        try:
            await asyncio.wait_for(page.close(), timeout=timeout)
        except Exception as e:
            logger.debug(f"Error closing page: {e}")
    if context:
        try:
            await asyncio.wait_for(context.close(), timeout=timeout)
        except Exception as e_ctx:
            if "Target page, context or browser has been closed" not in str(e_ctx):
                logger.debug(f"Error closing context: {e_ctx}")
    if browser and not profile_dir_usable:
        try:
            await asyncio.wait_for(browser.close(), timeout=timeout)
        except Exception as e_browser:
            logger.debug(f"Error closing browser: {e_browser}")
    await asyncio.sleep(timeout)
    killed = cleanup_playwright_processes()
    if killed:
        logger.info(f"Force killed {killed} lingering Playwright processes.")


JS_EXPAND_SHOWMORE_TWITTER = """
(maxClicks) => {
    let clicks = 0;
    const getButtons = () => Array.from(document.querySelectorAll('[role="button"]'))
        .filter(b => {
            const t = (b.textContent || '').toLowerCase();
            if (!t.includes('show more')) { return false; }
            const article = b.closest('article');
            if (!article) { return false; }
            const articleText = article.textContent || '';
            if (articleText.match(/grok/i)) { return false; }
            if (b.closest('[role="blockquote"]')) { return false; }
            return true;
        });

    while (clicks < maxClicks) {
        const buttonsToClick = getButtons();
        if (buttonsToClick.length === 0) break;
        const button = buttonsToClick[0];
        try {
            button.click();
            clicks++;
        } catch (e) {
            break;
        }
    }
    return clicks;
}
"""

JS_CLICK_SEE_MORE_GROUNDNEWS = """
(maxClicks) => {
    let clicks = 0;
    function findButton() {
        const byId = document.querySelector('#more-stories-my-feed');
        if (byId) return byId;
        return Array.from(document.querySelectorAll('button')).find(b =>
            (b.textContent || '').toLowerCase().includes('see more stories'));
    }
    while (clicks < maxClicks) {
        const btn = findButton();
        if (!btn) break;
        try {
            btn.click();
            clicks++;
        } catch (e) {
            break;
        }
    }
    return clicks;
}
"""

JS_EXTRACT_TWEETS_TWITTER = """
() => {
    const tweets = [];
    document.querySelectorAll('article[data-testid="tweet"]').forEach(article => {
        try {
            const timeTag = article.querySelector('time');
            const timestamp = timeTag ? timeTag.getAttribute('datetime') : null;

            let tweetLink = null, id = '', username = 'unknown_user';

            const primaryLinkElement = timeTag ? timeTag.closest('a[href*="/status/"]') : null;
            if (primaryLinkElement) {
                tweetLink = primaryLinkElement.href;
            } else {
                const articleLinks = Array.from(article.querySelectorAll('a[href*="/status/"]'));
                if (articleLinks.length > 0) {
                    tweetLink = articleLinks.find(link => !link.href.includes("/photo/") && !link.href.includes("/video/"))?.href || articleLinks[0].href;
                }
            }

            if (tweetLink) {
                const match = tweetLink.match(/\/([a-zA-Z0-9_]+)\/status\/(\d+)/);
                if (match) {
                    username = match[1];
                    id = match[2];
                }
            }

            const tweetTextElement = article.querySelector('div[data-testid="tweetText"]');
            const content = tweetTextElement ? tweetTextElement.innerText.trim() : '';

            const imageElements = article.querySelectorAll('div[data-testid="tweetPhoto"] img');
            const imageUrls = Array.from(imageElements).map(img => img.src);
            const altTexts = Array.from(imageElements).map(img => img.alt || null);

            const socialContextElement = article.querySelector('div[data-testid="socialContext"]');
            let is_repost = false, reposted_by = null;
            if (socialContextElement && /reposted|retweeted/i.test(socialContextElement.innerText)) {
                is_repost = true;
                const userLinkInContext = socialContextElement.querySelector('a[href^="/"]');
                if (userLinkInContext) {
                    const hrefParts = userLinkInContext.href.split('/');
                    reposted_by = hrefParts.filter(part =>
                        !['analytics', 'likes', 'media', 'status', 'with_replies', 'following', 'followers', ''].includes(part)
                    ).pop();
                }
            }

            if (content || imageUrls.length > 0 || article.querySelector('[data-testid="videoPlayer"]')) {
                tweets.push({
                    id: id || `no-id-${Date.now()}-${Math.random()}`,
                    username,
                    content,
                    timestamp: timestamp || new Date().toISOString(),
                    is_repost,
                    reposted_by,
                    tweet_url: tweetLink || (id ? `https://x.com/${username}/status/${id}` : ''),
                    image_urls: imageUrls,
                    alt_texts: altTexts
                });
            }
        } catch (e) {
            // console.warn('Error extracting individual tweet:', e);
        }
    });
    return tweets;
}
"""

async def _scrape_with_bs(url: str) -> Optional[str]:
    """Fallback scraping using aiohttp and BeautifulSoup with a Googlebot user agent."""
    logger.info(f"Attempting BeautifulSoup fallback for {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
    }
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=20) as resp:
                resp.raise_for_status()
                html = await resp.text()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            if len(text) > config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT:
                logger.info(
                    f"BS4 scraped content from {url} truncated from {len(text)} to {config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT} characters."
                )
                text = text[: config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT] + "..."
            return text
    except Exception as e:
        logger.warning(f"BeautifulSoup fallback failed for {url}: {e}")
    return None

async def scrape_website(
    url: str,
    progress_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    screenshots_dir: Optional[str] = None
) -> Tuple[Optional[str], List[str]]:
    if (
        url.startswith("https://www.nytimes.com/")
        or url.startswith("https://www.wsj.com/")
        or url.startswith("https://www.thedailybeast.com/")
    ):
        if "nytimes.com" in url:
            archive_domain = "NYTimes"
        elif "wsj.com" in url:
            archive_domain = "WSJ"
        else:
            archive_domain = "DailyBeast"
        url = f"https://archive.is/newest/{url}"
        logger.info(f"{archive_domain} URL detected. Using archive.is: {url}")
    logger.info(f"Attempting to scrape website: {url}")
    screenshot_paths: List[str] = []
    if screenshots_dir:
        try:
            os.makedirs(screenshots_dir, exist_ok=True)
            logger.info(f"Screenshots will be saved in: {screenshots_dir}")
        except OSError as e:
            logger.error(f"Could not create screenshots directory {screenshots_dir}: {e}. Screenshots will not be saved.")
            screenshots_dir = None # Disable screenshots if dir creation fails

    user_data_dir = os.path.join(os.getcwd(), ".pw-profile")
    profile_dir_usable = True
    if not os.path.exists(user_data_dir):
        try:
            os.makedirs(user_data_dir, exist_ok=True)
        except OSError as e:
            profile_dir_usable = False
            logger.error(f"Could not create or access .pw-profile directory: {e}. Playwright will use a non-persistent context.")

    context_manager: Optional[Any] = None
    browser_instance_sw: Optional[Any] = None
    page: Optional[Any] = None

    try:
        async with PLAYWRIGHT_SEM:
            async with async_playwright() as p:
                if profile_dir_usable:
                    context = await p.chromium.launch_persistent_context(
                        user_data_dir,
                        headless=config.HEADLESS_PLAYWRIGHT,
                        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    )
                else:
                    logger.warning("Using non-persistent context for scrape_website due to profile directory issue.")
                    browser_instance_sw = await p.chromium.launch(
                        headless=config.HEADLESS_PLAYWRIGHT,
                        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"]
                    )
                    context = await browser_instance_sw.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        java_script_enabled=True,
                        ignore_https_errors=True
                    )
                context_manager = context
                page = await context_manager.new_page()

                # Wait only until DOM content loads to avoid hanging on pages that never go network idle
                logger.info(f"Navigating to {url} and waiting for DOM content to load...")
                await page.goto(url, wait_until='domcontentloaded', timeout=35000)
                logger.info(
                    f"Navigation to {url} complete. Allowing time for additional JS rendering..."
                )
                await asyncio.sleep(2)  # Give 2 seconds for JS to potentially finish rendering after load

                if config.SCRAPE_SCROLL_ATTEMPTS > 0:
                    logger.info(
                        f"Scrolling page up to {config.SCRAPE_SCROLL_ATTEMPTS} times to load dynamic content."
                    )
                    try:
                        last_height = await page.evaluate("document.body.scrollHeight")
                    except Exception as e_eval:
                        logger.warning(f"Could not get initial scroll height for {url}: {e_eval}")
                        last_height = 0
                    for scroll_attempt in range(config.SCRAPE_SCROLL_ATTEMPTS):
                        if page.is_closed():
                            logger.warning(f"Playwright page closed unexpectedly while scrolling {url}.")
                            break
                        if screenshots_dir:
                            try:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                screenshot_filename = f"screenshot_scroll_{scroll_attempt + 1}_{timestamp}.png"
                                full_screenshot_path = os.path.join(screenshots_dir, screenshot_filename)
                                await page.screenshot(path=full_screenshot_path)
                                screenshot_paths.append(full_screenshot_path)
                                logger.info(f"Saved screenshot: {full_screenshot_path}")
                            except Exception as e_ss:
                                logger.error(f"Failed to take screenshot: {e_ss}")

                        try:
                            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        except Exception as e_scroll:
                            logger.warning(f"Error scrolling page {url}: {e_scroll}")
                            break
                        await asyncio.sleep(1.5)  # Wait for content to load after scroll
                        try:
                            new_height = await page.evaluate("document.body.scrollHeight")
                        except Exception as e_eval:
                            logger.warning(f"Could not get updated scroll height for {url}: {e_eval}")
                            break
                        if new_height == last_height:
                            break
                        last_height = new_height

                content_selectors = ["article", "main", "div[role='main']", "body"]
                content = ""
                for selector in content_selectors:
                    try:
                        element = page.locator(selector).first
                        if await element.count() > 0 and await element.is_visible(timeout=2000): # Shorter timeout for visibility check
                            logger.debug(f"Attempting to get text from selector: '{selector}'")
                            content = await element.inner_text(timeout=5000)
                            if content and len(content.strip()) > 200:
                                logger.info(f"Found substantial content with selector '{selector}'.")
                                break
                        else:
                            logger.debug(f"Selector '{selector}' not found or not visible on {url}")
                    except PlaywrightTimeoutError:
                        logger.debug(f"Timeout waiting for selector '{selector}' or its text on {url}")
                    except Exception as e_sel:
                        logger.warning(f"Error processing selector '{selector}' on {url}: {e_sel}")

                if (not content or len(content.strip()) < 100) and page.url != "about:blank":
                    logger.info(f"Primary selectors yielded little content for {url}. Falling back to document.body.innerText.")
                    try:
                        body_content = await page.evaluate('document.body ? document.body.innerText : ""')
                        if body_content:
                            content = body_content
                    except Exception as e_body_eval:
                        logger.warning(f"Error evaluating document.body.innerText for {url}: {e_body_eval}")

                if content:
                    content = content.strip()
                    content = re.sub(r"\s\s+", " ", content)
                    if len(content) > config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT:
                        logger.info(
                            f"Scraped content from {url} was truncated from {len(content)} to {config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT} characters."
                        )
                        content = (
                            content[: config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT] + "..."
                        )
                    return content if content else None, screenshot_paths
                else:
                    logger.warning(
                        f"No substantial content extracted from {url} after trying all selectors and body.innerText fallback."
                    )
                    bs_content = await _scrape_with_bs(url)
                    if bs_content:
                        logger.info("BeautifulSoup fallback succeeded.")
                        return bs_content, screenshot_paths # screenshot_paths will be empty here
                    return None, screenshot_paths # screenshot_paths will be empty here

    except PlaywrightTimeoutError:
        logger.error(f"Playwright timed out during the scraping process for {url}")
        if progress_callback: # Notify about timeout
            await progress_callback(f"Scraping {url} timed out.")
        return "Scraping timed out.", screenshot_paths # Return collected paths even on timeout
    except Exception as e:
        logger.error(f"Playwright encountered an unexpected error for {url}: {e}", exc_info=True)
        if progress_callback: # Notify about error
            await progress_callback(f"Failed to scrape {url} due to an error.")
        return "Failed to scrape the website due to an unexpected error.", screenshot_paths # Return collected paths on error
    finally:
        await _graceful_close_playwright(page, context_manager, browser_instance_sw, profile_dir_usable)

async def scrape_latest_tweets(username_queried: str, limit: int = 10, progress_callback: Optional[Callable[[str], Awaitable[None]]] = None) -> List[TweetData]:
    logger.info(f"Scraping last {limit} tweets for @{username_queried} (profile page, with replies) using Playwright JS execution.")
    tweets_collected: List[TweetData] = []
    seen_tweet_ids: set[str] = set()

    user_data_dir = os.path.join(os.getcwd(), ".pw-profile")
    profile_dir_usable = True
    if not os.path.exists(user_data_dir):
        try:
            os.makedirs(user_data_dir, exist_ok=True)
        except OSError:
            profile_dir_usable = False
            logger.error("Could not create .pw-profile. Using non-persistent context for tweet scraping.")

    context_manager: Optional[Any] = None
    browser_instance_st: Optional[Any] = None
    page: Optional[Any] = None
    try:
        async with PLAYWRIGHT_SEM:
            async with async_playwright() as p:
                if profile_dir_usable:
                    context = await p.chromium.launch_persistent_context(
                        user_data_dir, headless=config.HEADLESS_PLAYWRIGHT,
                        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
                        slow_mo=150
                    )
                else:
                    logger.warning("Using non-persistent context for tweet scraping.")
                    browser_instance_st = await p.chromium.launch(
                        headless=config.HEADLESS_PLAYWRIGHT,
                        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                        slow_mo=150
                    )
                    context = await browser_instance_st.new_context(
                        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
                    )
                context_manager = context
                page = await context_manager.new_page()

                url = f"https://x.com/{username_queried.lstrip('@')}/with_replies"
                logger.info(f"Navigating to Twitter profile: {url}")
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")

                try:
                    await page.wait_for_selector("article[data-testid='tweet']", timeout=30000)
                    logger.info("Initial tweet articles detected on page.")
                    await asyncio.sleep(1.5); await page.keyboard.press("Escape"); await asyncio.sleep(0.5); await page.keyboard.press("Escape")
                except PlaywrightTimeoutError:
                    logger.warning(f"Timed out waiting for initial tweet articles for @{username_queried}. Page might be empty or blocked."); return []

                max_scroll_attempts = limit + 15
                for scroll_attempt in range(max_scroll_attempts):
                    if len(tweets_collected) >= limit: break

                    try:
                        clicked_count = await page.evaluate(JS_EXPAND_SHOWMORE_TWITTER, 5)
                        if clicked_count > 0:
                            logger.info(f"Clicked {clicked_count} 'Show more' elements on Twitter page.");
                            await asyncio.sleep(1.5 + random.uniform(0.3, 0.9))
                    except PlaywrightTimeoutError: # More specific error for JS execution timeout
                        logger.warning(f"Timeout during JS 'Show More' execution for @{username_queried}.")
                    except Exception as e_sm:
                        logger.warning(f"JavaScript 'Show More' execution error: {e_sm}")

                    extracted_this_round: List[Dict[str, Any]] = []
                    newly_added_count = 0
                    try:
                        extracted_this_round = await page.evaluate(JS_EXTRACT_TWEETS_TWITTER)
                    except PlaywrightTimeoutError: # More specific error for JS execution timeout
                        logger.warning(f"Timeout during JS tweet extraction for @{username_queried}.")
                    except Exception as e_js:
                        logger.error(f"JavaScript tweet extraction (JS_EXTRACT_TWEETS_TWITTER) error: {e_js}")

                    for data in extracted_this_round:
                        uid_parts = [
                            str(data.get('id', '')),
                            str(data.get("username","")),
                            str(data.get("content") or "")[:30],
                            str(data.get("timestamp",""))
                        ]
                        uid = hashlib.md5("".join(filter(None, uid_parts)).encode('utf-8')).hexdigest()


                        if uid and uid not in seen_tweet_ids:
                            tweet = TweetData(
                                id=data.get('id', ''),
                                username=data.get('username', 'unknown_user'),
                                content=data.get('content', ''),
                                timestamp=data.get('timestamp', ''),
                                tweet_url=data.get('tweet_url', ''),
                                is_repost=data.get('is_repost', False),
                                reposted_by=data.get('reposted_by'),
                                image_urls=data.get('image_urls', []),
                                alt_texts=data.get('alt_texts', [])
                            )
                            tweets_collected.append(tweet)
                            seen_tweet_ids.add(uid)
                            newly_added_count +=1
                            if progress_callback:
                                try:
                                    await progress_callback(f"Scraped {len(tweets_collected)}/{limit} tweets for @{username_queried}...")
                                except Exception as e_cb:
                                    logger.warning(f"Progress callback error: {e_cb}")
                            if len(tweets_collected) >= limit: break

                    if newly_added_count == 0 and scroll_attempt > (limit // 2 + 7): # Increased patience slightly
                        logger.info("No new unique tweets found in several scroll attempts. Stopping.")
                        if progress_callback:
                            await progress_callback(f"Stopping early: No new unique tweets found after {scroll_attempt + 1} scrolls. Collected {len(tweets_collected)}.")
                        break

                    # Scroll down to load more tweets
                    try:
                        await page.evaluate("window.scrollBy(0, window.innerHeight * 1.5);")
                        await asyncio.sleep(random.uniform(3.0, 5.0))
                    except PlaywrightTimeoutError:
                        logger.warning(f"Timeout during page scroll for @{username_queried}. Assuming end of content or page issue.")
                        break # Stop scrolling if it times out

    except PlaywrightTimeoutError as e:
        logger.warning(f"Playwright overall timeout during tweet scraping for @{username_queried}: {e}")
        if progress_callback:
            await progress_callback(f"Tweet scraping for @{username_queried} timed out overall. Collected {len(tweets_collected)}.")
    except Exception as e:
        logger.error(f"Unexpected error during tweet scraping for @{username_queried}: {e}", exc_info=True)
        if progress_callback:
            await progress_callback(f"An unexpected error occurred while scraping tweets for @{username_queried}. Collected {len(tweets_collected)}.")
    finally:
        await _graceful_close_playwright(
            page,
            context_manager,
            browser_instance_st,
            profile_dir_usable,
        )

    tweets_collected.sort(key=lambda x: x.timestamp, reverse=True)
    logger.info(f"Finished scraping. Collected {len(tweets_collected)} unique tweets for @{username_queried}.")
    return tweets_collected[:limit]


async def scrape_home_timeline(limit: int = 10, progress_callback: Optional[Callable[[str], Awaitable[None]]] = None) -> List[TweetData]:
    """Scrape tweets from the logged-in home timeline on X/Twitter."""
    logger.info(f"Scraping last {limit} tweets from the home timeline using Playwright JS execution.")

    tweets_collected: List[TweetData] = []
    seen_tweet_ids: set[str] = set()

    user_data_dir = os.path.join(os.getcwd(), ".pw-profile")
    profile_dir_usable = True
    if not os.path.exists(user_data_dir):
        try:
            os.makedirs(user_data_dir, exist_ok=True)
        except OSError:
            profile_dir_usable = False
            logger.error("Could not create .pw-profile. Using non-persistent context for tweet scraping.")

    context_manager: Optional[Any] = None
    browser_instance_st: Optional[Any] = None
    page: Optional[Any] = None
    try:
        async with PLAYWRIGHT_SEM:
            async with async_playwright() as p:
                if profile_dir_usable:
                    context = await p.chromium.launch_persistent_context(
                        user_data_dir,
                        headless=config.HEADLESS_PLAYWRIGHT,
                        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
                        slow_mo=150,
                    )
                else:
                    logger.warning("Using non-persistent context for tweet scraping.")
                    browser_instance_st = await p.chromium.launch(
                        headless=config.HEADLESS_PLAYWRIGHT,
                        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                        slow_mo=150,
                    )
                    context = await browser_instance_st.new_context(
                        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
                    )
                context_manager = context
                page = await context_manager.new_page()

                url = "https://x.com/home"
                logger.info(f"Navigating to Twitter home timeline: {url}")
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")

                try:
                    await page.wait_for_selector("article[data-testid='tweet']", timeout=30000)
                    logger.info("Initial tweet articles detected on home timeline.")
                    await asyncio.sleep(1.5)
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(0.5)
                    await page.keyboard.press("Escape")
                except PlaywrightTimeoutError:
                    logger.warning("Timed out waiting for initial tweet articles on home timeline. Page might be empty or blocked.")
                    return []

                max_scroll_attempts = limit + 15
                for scroll_attempt in range(max_scroll_attempts):
                    if len(tweets_collected) >= limit:
                        break

                    try:
                        clicked_count = await page.evaluate(JS_EXPAND_SHOWMORE_TWITTER, 5)
                        if clicked_count > 0:
                            logger.info(f"Clicked {clicked_count} 'Show more' elements on Twitter page.")
                            await asyncio.sleep(1.5 + random.uniform(0.3, 0.9))
                    except PlaywrightTimeoutError:
                        logger.warning("Timeout during JS 'Show More' execution on home timeline.")
                    except Exception as e_sm:
                        logger.warning(f"JavaScript 'Show More' execution error on home timeline: {e_sm}")

                    extracted_this_round: List[Dict[str, Any]] = []
                    newly_added_count = 0
                    try:
                        extracted_this_round = await page.evaluate(JS_EXTRACT_TWEETS_TWITTER)
                    except PlaywrightTimeoutError:
                        logger.warning("Timeout during JS tweet extraction on home timeline.")
                    except Exception as e_js:
                        logger.error(f"JavaScript tweet extraction error on home timeline: {e_js}")

                    for data in extracted_this_round:
                        uid_parts = [
                            str(data.get("id", "")),
                            str(data.get("username", "")),
                            str(data.get("content") or "")[:30],
                            str(data.get("timestamp", "")),
                        ]
                        uid = hashlib.md5("".join(filter(None, uid_parts)).encode("utf-8")).hexdigest()

                        if uid and uid not in seen_tweet_ids:
                            tweet = TweetData(
                                id=data.get('id', ''),
                                username=data.get('username', 'unknown_user'),
                                content=data.get('content', ''),
                                timestamp=data.get('timestamp', ''),
                                tweet_url=data.get('tweet_url', ''),
                                is_repost=data.get('is_repost', False),
                                reposted_by=data.get('reposted_by'),
                                image_urls=data.get('image_urls', []),
                                alt_texts=data.get('alt_texts', [])
                            )
                            tweets_collected.append(tweet)
                            seen_tweet_ids.add(uid)
                            newly_added_count += 1
                            if progress_callback:
                                try:
                                    await progress_callback(f"Scraped {len(tweets_collected)}/{limit} tweets from home timeline...")
                                except Exception as e_cb:
                                    logger.warning(f"Progress callback error: {e_cb}")
                            if len(tweets_collected) >= limit:
                                break

                    if newly_added_count == 0 and scroll_attempt > (limit // 2 + 7):
                        logger.info("No new unique tweets found in several scroll attempts. Stopping.")
                        if progress_callback:
                            await progress_callback(
                                f"Stopping early: No new unique tweets found after {scroll_attempt + 1} scrolls. Collected {len(tweets_collected)}."
                            )
                        break

                    try:
                        await page.evaluate("window.scrollBy(0, window.innerHeight * 1.5);")
                        await asyncio.sleep(random.uniform(3.0, 5.0))
                    except PlaywrightTimeoutError:
                        logger.warning("Timeout during page scroll on home timeline. Assuming end of content or page issue.")
                        break

    except PlaywrightTimeoutError as e:
        logger.warning(f"Playwright overall timeout during tweet scraping for home timeline: {e}")
        if progress_callback:
            await progress_callback(
                f"Tweet scraping for home timeline timed out overall. Collected {len(tweets_collected)}."
            )
    except Exception as e:
        logger.error(f"Unexpected error during tweet scraping for home timeline: {e}", exc_info=True)
        if progress_callback:
            await progress_callback(
                f"An unexpected error occurred while scraping tweets for home timeline. Collected {len(tweets_collected)}."
            )
    finally:
        await _graceful_close_playwright(
            page,
            context_manager,
            browser_instance_st,
            profile_dir_usable,
        )

    tweets_collected.sort(key=lambda x: x.timestamp, reverse=True)
    logger.info(f"Finished scraping. Collected {len(tweets_collected)} unique tweets from home timeline.")
    return tweets_collected[:limit]


async def query_searx(query: str) -> List[Dict[str, Any]]:
    logger.info(f"Querying Searx for: '{query}'")
    params: Dict[str, Any] = {'q': query, 'format': 'json', 'language': 'en-US'}
    if config.SEARX_PREFERENCES:
        if "%s" in config.SEARX_PREFERENCES:
             params['preferences'] = config.SEARX_PREFERENCES % query
        else:
             params['preferences'] = config.SEARX_PREFERENCES

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(config.SEARX_URL, params=params, timeout=10) as response:
                response.raise_for_status()
                results_json = await response.json()
                return results_json.get('results', [])[:5]
    except aioh_utils.pyttp.ClientError as e:
        logger.error(f"Searx query failed for '{query}': {e}")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from Searx for query '{query}'")
    return []

async def fetch_youtube_transcript(url: str) -> Optional[str]:
    try:
        video_id_match = re.search(r'(?:v=|\/|embed\/|shorts\/|youtu\.be\/|googleusercontent\.com\/youtube\.com\/(?:[0-8]\/)?)([0-9A-Za-z_-]{11})', url)

        if not video_id_match:
            logger.warning(f"No YouTube video ID found in URL: {url}")
            return None

        video_id = video_id_match.group(1)
        logger.info(f"Fetching YouTube transcript for video ID: {video_id} (from URL: {url})")

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript_obj: Optional[Any] = None

        try:
            transcript_obj = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
        except NoTranscriptFound:
            try:
                transcript_obj = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
            except NoTranscriptFound:
                available_langs = [t.language for t in transcript_list._transcripts.values()]
                if available_langs:
                    logger.warning(f"No English transcript for {video_id}. Available: {available_langs}. Trying first available: {available_langs[0]}.")
                    try:
                        transcript_obj = transcript_list.find_generated_transcript([available_langs[0]])
                    except NoTranscriptFound:
                         logger.warning(f"Could not fetch even the first available language transcript for {video_id}.")
                else:
                    logger.warning(f"No English or any other language transcripts found for {video_id}.")

        if transcript_obj:
            fetched_data = transcript_obj.fetch()
            # The object returned by fetch() is a FetchedTranscript, which is iterable
            # and yields FetchedTranscriptSnippet objects.
            # Each snippet object has a .text attribute.
            full_text = " ".join([entry.text for entry in fetched_data])

            if len(full_text) > config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT:
                logger.info(f"YouTube transcript for {url} (ID: {video_id}) truncated from {len(full_text)} to {config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT} chars.")
                full_text = full_text[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT] + "..."

            return f"(Language: {transcript_obj.language}) {full_text}" if transcript_obj.language != 'en' and not transcript_obj.language.startswith('en-') else full_text
        else:
            logger.warning(f"No transcript could be fetched for YouTube video: {url} (ID: {video_id})")
            return None

    except xml.etree.ElementTree.ParseError as e_xml:
        video_id_for_log = video_id_match.group(1) if 'video_id_match' in locals() and video_id_match else 'unknown_id_xml_err'
        logger.error(f"Failed to parse YouTube transcript XML for {url} (ID: {video_id_for_log}): {e_xml}", exc_info=True)
        return None
    except Exception as e:
        video_id_for_log = video_id_match.group(1) if 'video_id_match' in locals() and video_id_match else 'unknown_id_gen_err'
        logger.error(f"Failed to fetch YouTube transcript for {url} (ID: {video_id_for_log}): {e}", exc_info=True)
        return None

async def fetch_rss_entries(feed_url: str) -> List[Dict[str, Any]]:
    """Fetch and parse RSS feed entries.

    Each returned entry includes ``pubDate_dt`` as a timezone-aware
    ``datetime`` for easier sorting and display.
    """
    logger.info(f"Fetching RSS feed: {feed_url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(feed_url, timeout=15) as resp:
                resp.raise_for_status()
                text = await resp.text()
    except Exception as e:
        logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
        return []

    try:
        root = xml.etree.ElementTree.fromstring(text)
        channel = root.find("channel") or root
        items = channel.findall("item")
        entries: List[Dict[str, Any]] = []
        one_days_ago = datetime.now(timezone.utc) - timedelta(days=1)
        for it in items:
            pub_date_str = it.findtext("pubDate") or ""
            pub_date_dt = None
            if pub_date_str:
                try:
                    pub_date_dt = parsedate_to_datetime(pub_date_str)
                    if pub_date_dt.tzinfo is None:
                        pub_date_dt = pub_date_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    pub_date_dt = None

            if pub_date_dt is None or pub_date_dt < one_days_ago:
                continue

            link_url = it.findtext("link") or ""
            if link_url.startswith("https://www.cbsnews.com/video/"):
                continue
            if (
                link_url.startswith("https://www.cbsnews.com/")
                and not link_url.startswith("https://www.cbsnews.com/news/")
            ):
                continue

            entries.append({
                "title": it.findtext("title") or "",
                "link": link_url,
                "guid": it.findtext("guid") or link_url or "",
                "pubDate": pub_date_str,
                "pubDate_dt": pub_date_dt,
                "description": it.findtext("description") or "",
            })
        return entries
    except Exception as e:
        logger.error(f"Failed to parse RSS feed {feed_url}: {e}")
        return []


async def scrape_ground_news_my(limit: int = 10) -> List[GroundNewsArticle]:
    """Scrape the Ground News 'My Feed' page for article links.

    Requires that the user is already logged in within the persistent
    Playwright profile (``.pw-profile``). If not logged in, this function
    will likely return an empty list.

    The scraper searches for any link containing ``See the Story`` and
    attempts to find a nearby title element. It scrolls the page
    gradually between attempts to allow dynamic content to load.

    Parameters
    ----------
    limit : int, optional
        Maximum number of article links to return, by default ``10``.

    Returns
    -------
    List[GroundNewsArticle]
        Parsed article entries with title and URL.
    """
    logger.info("Scraping Ground News 'My Feed' for articles...")
    articles: List[GroundNewsArticle] = []

    user_data_dir = os.path.join(os.getcwd(), ".pw-profile")
    profile_dir_usable = True
    if not os.path.exists(user_data_dir):
        try:
            os.makedirs(user_data_dir, exist_ok=True)
            logger.info("Created .pw-profile directory for Ground News scrape.")
        except OSError as exc:
            logger.error("Could not create .pw-profile directory: %s", exc)
            profile_dir_usable = False

    context_manager: Optional[Any] = None
    browser_instance: Optional[Any] = None
    page: Optional[Any] = None

    try:
        async with PLAYWRIGHT_SEM:
            async with async_playwright() as p:
                if profile_dir_usable:
                    context = await p.chromium.launch_persistent_context(
                        user_data_dir,
                        headless=config.HEADLESS_PLAYWRIGHT,
                        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                        ignore_https_errors=True,
                    )
                else:
                    browser_instance = await p.chromium.launch(
                        headless=config.HEADLESS_PLAYWRIGHT,
                        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
                    )
                    context = await browser_instance.new_context(ignore_https_errors=True)
                context_manager = context
                page = await context_manager.new_page()

                await page.goto("https://ground.news/my", wait_until="domcontentloaded")
                await asyncio.sleep(5)

                try:
                    clicked = await page.evaluate(
                        JS_CLICK_SEE_MORE_GROUNDNEWS,
                        config.GROUND_NEWS_SEE_MORE_CLICKS,
                    )
                    if clicked:
                        logger.info(
                            "Clicked 'See more stories' button %s time(s) on Ground News",
                            clicked,
                        )
                        await asyncio.sleep(2)
                except Exception as e_click:
                    logger.debug(
                        "Error clicking 'See more stories' on Ground News: %s",
                        e_click,
                    )

                seen_urls: Set[str] = set()
                scroll_attempt = 0
                while len(articles) < limit and scroll_attempt <= config.SCRAPE_SCROLL_ATTEMPTS:
                    extracted = await page.evaluate(
                        """
                        () => {
                            const arts = [];
                            document.querySelectorAll('a').forEach(link => {
                                if (link.textContent && link.textContent.includes('See the Story')) {
                                    const container = link.closest('article, div') || link.parentElement;
                                    const titleEl = container ? container.querySelector('h3, h2, h4') : null;
                                    const title = titleEl ? titleEl.textContent.trim() : link.textContent.trim();
                                    arts.push({title, url: link.href});
                                }
                            });
                            return arts;
                        }
                        """
                    )
                    for item in extracted:
                        if not isinstance(item, dict):
                            continue
                        title = str(item.get("title", "")).strip()
                        url = str(item.get("url", "")).strip()
                        if title and url and url not in seen_urls:
                            seen_urls.add(url)
                            articles.append(GroundNewsArticle(title=title, url=url))
                            if len(articles) >= limit:
                                break
                    if len(articles) >= limit:
                        break
                    try:
                        await page.evaluate("window.scrollBy(0, window.innerHeight)")
                        await asyncio.sleep(2)
                    except Exception as e_scroll:
                        logger.warning("Scrolling failed on Ground News page: %s", e_scroll)
                        break
                    scroll_attempt += 1
    except Exception as e:
        logger.error("Error scraping Ground News: %s", e, exc_info=True)
    finally:
        await _graceful_close_playwright(page, context_manager, browser_instance, profile_dir_usable)

    logger.info("Found %s articles on Ground News", len(articles))
    return articles
