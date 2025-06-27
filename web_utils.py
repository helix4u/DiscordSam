import asyncio
import logging
import os
import re
import json
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
import random
import hashlib

import aiohttp
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError # type: ignore
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound # type: ignore
import xml.etree.ElementTree

# Assuming config is imported from config.py
from config import config

logger = logging.getLogger(__name__)

PLAYWRIGHT_SEM = asyncio.Semaphore(config.PLAYWRIGHT_MAX_CONCURRENCY)

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

            if (content || article.querySelector('[data-testid="tweetPhoto"], [data-testid="videoPlayer"]')) {
                tweets.push({
                    id: id || `no-id-${Date.now()}-${Math.random()}`,
                    username,
                    content,
                    timestamp: timestamp || new Date().toISOString(),
                    is_repost,
                    reposted_by,
                    tweet_url: tweetLink || (id ? `https://x.com/${username}/status/${id}` : '')
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

async def scrape_website(url: str) -> Optional[str]:
    logger.info(f"Attempting to scrape website: {url}")
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

                # Determine the best container selector first
                content_selectors = ["article", "main", "div[role='main']", "body"]
                main_container_selector = "body" # Default
                initial_content_check = ""

                for selector_attempt in content_selectors:
                    try:
                        element = page.locator(selector_attempt).first
                        if await element.count() > 0 and await element.is_visible(timeout=1500):
                            initial_content_check = await element.inner_text(timeout=3000)
                            if initial_content_check and len(initial_content_check.strip()) > 100:
                                main_container_selector = selector_attempt
                                logger.info(f"Using '{main_container_selector}' as main container for additive scraping.")
                                break
                            else:
                                logger.debug(f"Selector '{selector_attempt}' content too short or not found.")
                        else:
                            logger.debug(f"Selector '{selector_attempt}' not found or not visible on {url}")
                    except PlaywrightTimeoutError:
                        logger.debug(f"Timeout checking selector '{selector_attempt}' on {url}")
                    except Exception as e_sel_check:
                        logger.warning(f"Error checking selector '{selector_attempt}' on {url}: {e_sel_check}")

                logger.info(f"Final main container selector for JS: '{main_container_selector}'")

                collected_texts_hashes = set()
                collected_texts_ordered = []
                total_collected_length = 0

                if config.SCRAPE_SCROLL_ATTEMPTS > 0:
                    logger.info(
                        f"Scrolling page up to {config.SCRAPE_SCROLL_ATTEMPTS} times to load dynamic content, extracting additively."
                    )
                    last_height = await page.evaluate("document.body.scrollHeight")

                    for i in range(config.SCRAPE_SCROLL_ATTEMPTS):
                        # Extract content blocks using JS
                        try:
                            current_blocks = await page.evaluate(JS_EXTRACT_TEXT_BLOCKS, main_container_selector)
                            newly_added_count_this_scroll = 0
                            for block_text in current_blocks:
                                block_hash = hashlib.md5(block_text.encode('utf-8')).hexdigest()
                                if block_hash not in collected_texts_hashes:
                                    if (total_collected_length + len(block_text)) <= config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT:
                                        collected_texts_hashes.add(block_hash)
                                        collected_texts_ordered.append(block_text)
                                        total_collected_length += len(block_text) + 2 # +2 for \n\n
                                        newly_added_count_this_scroll +=1
                                    else:
                                        logger.info(f"Reached max text length limit ({config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT}), stopping further text collection.")
                                        break
                            logger.info(f"Scroll {i+1}: Added {newly_added_count_this_scroll} new text blocks. Total blocks: {len(collected_texts_ordered)}")
                            if total_collected_length >= config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT:
                                break # Break scroll loop if max length reached
                        except PlaywrightTimeoutError:
                            logger.warning(f"Timeout during JS text block extraction on scroll {i+1} for {url}.")
                        except Exception as e_js_extract:
                            logger.error(f"JS text block extraction error on scroll {i+1} for {url}: {e_js_extract}")

                        # Scroll down
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await asyncio.sleep(random.uniform(1.8, 2.5)) # Slightly longer, randomized sleep

                        new_height = await page.evaluate("document.body.scrollHeight")
                        if new_height == last_height and newly_added_count_this_scroll == 0 : # Stop if no new content and no new height
                            logger.info(f"Scroll {i+1}: Page height did not change and no new blocks added. Stopping scroll.")
                            break
                        last_height = new_height
                else: # No scrolling, just extract once using the JS block extractor
                    try:
                        current_blocks = await page.evaluate(JS_EXTRACT_TEXT_BLOCKS, main_container_selector)
                        for block_text in current_blocks:
                            block_hash = hashlib.md5(block_text.encode('utf-8')).hexdigest()
                            # No need to check hash here as it's the first and only pull
                            if (total_collected_length + len(block_text)) <= config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT:
                                collected_texts_hashes.add(block_hash) # Still add for consistency if this path is taken
                                collected_texts_ordered.append(block_text)
                                total_collected_length += len(block_text) + 2
                            else:
                                break
                        logger.info(f"No-scroll extraction: Added {len(collected_texts_ordered)} text blocks.")
                    except Exception as e_js_extract_no_scroll:
                        logger.error(f"JS text block extraction error (no scroll) for {url}: {e_js_extract_no_scroll}")


                final_content = ""
                if collected_texts_ordered:
                    final_content = "\n\n".join(collected_texts_ordered)
                    # Redundant length check, but good for safety. The truncation is now primarily managed during collection.
                    if len(final_content) > config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT:
                        final_content = final_content[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT] + "..."
                        logger.info(f"Final collected content for {url} was slightly trimmed to ensure limit.")
                    return final_content if final_content.strip() else None

                # Fallback if JS block extraction yielded nothing, try original body.innerText or BS4
                logger.warning(f"Additive JS block extraction yielded no content for {url}. Trying fallbacks.")
                try:
                    body_content = await page.evaluate('document.body ? document.body.innerText : ""')
                    if body_content:
                        body_content = body_content.strip()
                        body_content = re.sub(r"\s\s+", " ", body_content)
                        if len(body_content) > config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT:
                             body_content = body_content[:config.MAX_SCRAPED_TEXT_LENGTH_FOR_PROMPT] + "..."
                        logger.info(f"Fallback to body.innerText succeeded for {url}.")
                        return body_content
                except Exception as e_body_eval:
                    logger.warning(f"Error evaluating document.body.innerText fallback for {url}: {e_body_eval}")

                # Final fallback to BeautifulSoup if all else fails
                bs_content = await _scrape_with_bs(url)
                if bs_content:
                    logger.info(f"Final fallback to BeautifulSoup succeeded for {url}.")
                    return bs_content

                logger.warning(f"No content extracted from {url} after all methods.")
                return None

    except PlaywrightTimeoutError:
        logger.error(f"Playwright timed out during the scraping process for {url}")
        return "Scraping timed out."
    except Exception as e:
        logger.error(f"Playwright encountered an unexpected error for {url}: {e}", exc_info=True)
        return "Failed to scrape the website due to an unexpected error."
    finally:
        if page and not page.is_closed():
            try: await page.close()
            except Exception: pass
        if context_manager:
            try: await context_manager.close()
            except Exception as e_ctx:
                if "Target page, context or browser has been closed" not in str(e_ctx):
                    logger.debug(f"Ignoring error during context closure for {url}: {e_ctx}")
        if browser_instance_sw and not profile_dir_usable:
            try: await browser_instance_sw.close()
            except Exception: pass

async def scrape_latest_tweets(username_queried: str, limit: int = 10) -> List[Dict[str, Any]]:
    logger.info(f"Scraping last {limit} tweets for @{username_queried} (profile page, with replies) using Playwright JS execution.")
    tweets_collected: List[Dict[str, Any]] = []
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
                            tweets_collected.append(data)
                            seen_tweet_ids.add(uid)
                            newly_added_count +=1
                            if len(tweets_collected) >= limit: break

                    if newly_added_count == 0 and scroll_attempt > (limit // 2 + 7):
                        logger.info("No new unique tweets found in several scroll attempts. Stopping.")
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
    except Exception as e:
        logger.error(f"Unexpected error during tweet scraping for @{username_queried}: {e}", exc_info=True)
    finally:
        if page and not page.is_closed():
            try: await page.close()
            except Exception as e_page_close_final: logger.debug(f"Ignoring error closing page (final) for @{username_queried}: {e_page_close_final}")
        if context_manager:
            try: await context_manager.close()
            except Exception as e_ctx_final:
                if "Target page, context or browser has been closed" not in str(e_ctx_final):
                    logger.debug(f"Error closing context (final) for @{username_queried}: {e_ctx_final}")
        if browser_instance_st and not profile_dir_usable:
            try: await browser_instance_st.close()
            except Exception as e_browser_final: logger.debug(f"Ignoring error closing browser (final) for @{username_queried}: {e_browser_final}")

    tweets_collected.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    logger.info(f"Finished scraping. Collected {len(tweets_collected)} unique tweets for @{username_queried}.")
    return tweets_collected[:limit]

JS_EXTRACT_TEXT_BLOCKS = """
(containerSelector) => {
    const blocks = [];
    const container = document.querySelector(containerSelector);
    if (!container) {
        return blocks;
    }

    // Attempt to find meaningful blocks of text.
    // This could be paragraphs, list items, or divs that are direct children
    // and seem to contain distinct pieces of information.
    // For simplicity, let's start with direct children <p> and <div> tags.
    // More sophisticated selectors might be needed for complex pages.
    // We also filter out very short text blocks.
    const MIN_BLOCK_LENGTH = 50; // Minimum characters for a block to be considered

    // Prioritize more specific elements if they exist, otherwise broader ones.
    let elements = Array.from(container.querySelectorAll('article, div[role="listitem"], li'));
    if (elements.length === 0) {
        elements = Array.from(container.querySelectorAll('p, div:not(:has(div))')); // divs that don't contain other divs
    }
    if (elements.length === 0 && container.children.length > 0) {
        elements = Array.from(container.children); // Fallback to direct children
    }


    elements.forEach(el => {
        const text = el.innerText ? el.innerText.trim() : '';
        if (text.length >= MIN_BLOCK_LENGTH) {
            // Basic duplicate check based on text content within this current extraction
            // More robust deduplication will happen in Python using hashes.
            if (!blocks.some(b => b.includes(text) || text.includes(b))) {
                 blocks.push(text);
            }
        }
    });
    return blocks;
}
"""

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
