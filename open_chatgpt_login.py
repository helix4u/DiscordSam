import asyncio
import os
from typing import Optional

from playwright.async_api import BrowserContext, async_playwright
from config import config


async def open_chatgpt_login():
    """
    Launch a browser with a persistent profile and keep it open
    until the user stops the script (Ctrl+C). Login cookies
    will be stored in .pw-profile for reuse.
    """
    user_data_dir = os.path.join(os.getcwd(), ".pw-profile")
    if not os.path.exists(user_data_dir):
        try:
            os.makedirs(user_data_dir)
            print(f"Created profile directory: {user_data_dir}")
        except OSError as e:
            print(f"Could not create .pw-profile directory: {e}")
            return

    print(f"Using profile directory: {user_data_dir}")

    async with async_playwright() as p:
        browser_context: Optional[BrowserContext] = None
        try:
            # Import the helper function from web_utils if available
            try:
                from web_utils import _get_playwright_args
                playwright_args = _get_playwright_args()
            except ImportError:
                # Fallback if web_utils isn't available
                playwright_args = ["--disable-blink-features=AutomationControlled"]
                if not config.HEADLESS_PLAYWRIGHT:
                    playwright_args.extend([
                        "--disable-background-networking",
                        "--disable-background-timer-throttling",
                        "--disable-backgrounding-occluded-windows",
                        "--disable-renderer-backgrounding",
                        "--disable-features=TranslateUI",
                        "--disable-infobars",
                        "--disable-notifications",
                    ])
            
            browser_context = await p.chromium.launch_persistent_context(
                user_data_dir,
                headless=config.HEADLESS_PLAYWRIGHT,
                args=playwright_args,
                slow_mo=100,
            )

            page = await browser_context.new_page()
            print("Navigating to ChatGPT (chat.openai.com)...")
            await page.goto(
                "https://chat.openai.com/",
                wait_until="networkidle",
                timeout=1_200_000,
            )

            print(
                "\n---------------------------------------------------------------------\n"
                "Browser is open. Log in if prompted. Your session will persist in\n"
                f"'{user_data_dir}'.\n"
                "Leave this terminal running; press Ctrl+C when you are finished.\n"
                "---------------------------------------------------------------------"
            )

            # Keep running until the user interrupts (Ctrl+C)
            await asyncio.Event().wait()

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Closing browser context...")

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            if browser_context:
                try:
                    await browser_context.close()
                except Exception as e_close:
                    print(f"Error closing browser context: {e_close}")


if __name__ == "__main__":
    print("Starting script to open ChatGPT for login...")
    asyncio.run(open_chatgpt_login())
    print("Script finished.")
