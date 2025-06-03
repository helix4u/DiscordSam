import asyncio
import os
from typing import Optional

from playwright.async_api import BrowserContext, async_playwright

async def open_chatgpt_login():
    """
    Launches a non-headless browser using a persistent context,
    navigates to the ChatGPT login page, and waits for the user to close it.
    This helps save the login session in the .pw-profile directory.
    """
    user_data_dir = os.path.join(os.getcwd(), ".pw-profile")
    profile_dir_exists_or_created = False
    if not os.path.exists(user_data_dir):
        try:
            os.makedirs(user_data_dir)
            print(f"Created profile directory: {user_data_dir}")
            profile_dir_exists_or_created = True
        except OSError as e:
            print(f"Could not create .pw-profile directory: {e}. Session might not persist across runs.")
            # No fallback to temporary profile, as the goal is persistence
    else:
        profile_dir_exists_or_created = True

    print(f"Attempting to use profile directory: {user_data_dir if profile_dir_exists_or_created else 'Profile directory not available'}")

    if not profile_dir_exists_or_created:
        print("Cannot proceed without a profile directory. Please ensure the directory can be created.")
        return

    async with async_playwright() as p:
        browser_context: Optional[BrowserContext] = None
        try:
            # Using launch_persistent_context to save login state
            browser_context = await p.chromium.launch_persistent_context(
                user_data_dir,
                headless=False,  # Makes the browser visible
                args=[
                    "--disable-blink-features=AutomationControlled",
                    # "--no-sandbox", # Uncomment if you run into sandbox issues, common in some environments
                    # "--disable-dev-shm-usage" # Can help in resource-constrained environments
                ],
                slow_mo=100  # Slows down Playwright operations to make it easier to see
            )
            
            page = await browser_context.new_page()
            
            print("Navigating to ChatGPT (chat.openai.com)...")
            # Navigate to the main chat page, which will redirect to login if not authenticated
            await page.goto("https://chat.openai.com/", wait_until="networkidle", timeout=120000)
            
            print("\n-------------------------------------------------------------------------")
            print("Browser is open. Please log in to ChatGPT in this window if prompted.")
            print("Once you have logged in and the main chat interface has loaded,")
            print("you can close the browser window.")
            print("Your login session should be saved in the '.pw-profile' directory.")
            print("-------------------------------------------------------------------------")
            
            # Keep the script running until the browser context is closed by the user
            await browser_context.wait_for_event("close")
            print("Browser closed by user. Session (if login was successful) should be saved.")

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
