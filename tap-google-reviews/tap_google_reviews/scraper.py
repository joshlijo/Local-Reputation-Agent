"""Playwright-based Google Maps reviews scraper with stealth and warm-up navigation."""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from dateutil.relativedelta import relativedelta
from playwright.sync_api import sync_playwright, Page

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def generate_review_id(reviewer_name: str, review_date: str, review_text: str | None) -> str:
    """Create a stable review ID from reviewer name, date, and first 50 chars of text."""
    text_part = (review_text or "")[:50]
    raw = f"{reviewer_name}|{review_date}|{text_part}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_relative_date(relative: str, now: datetime | None = None) -> str:
    """Convert relative date string (e.g. '2 weeks ago') to ISO 8601."""
    if now is None:
        now = datetime.now(IST)

    relative = relative.strip().lower()
    relative = re.sub(r"\ban?\b", "1", relative)

    match = re.search(r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", relative)
    if not match:
        logger.warning("Could not parse relative date: %s", relative)
        return now.isoformat()

    amount = int(match.group(1))
    unit = match.group(2)

    if unit == "second":
        dt = now - timedelta(seconds=amount)
    elif unit == "minute":
        dt = now - timedelta(minutes=amount)
    elif unit == "hour":
        dt = now - timedelta(hours=amount)
    elif unit == "day":
        dt = now - timedelta(days=amount)
    elif unit == "week":
        dt = now - timedelta(weeks=amount)
    elif unit == "month":
        dt = now - relativedelta(months=amount)
    elif unit == "year":
        dt = now - relativedelta(years=amount)
    else:
        dt = now

    return dt.isoformat()


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class GoogleReviewsScraper:
    """Scrapes Google Maps reviews using Playwright with stealth."""

    def __init__(
        self,
        google_maps_url: str,
        place_query: str | None = None,
        headless: bool = True,
        max_pages: int = 100,
        rate_limit_seconds: float = 1.0,
    ) -> None:
        self.google_maps_url = google_maps_url
        self.place_query = place_query
        self.headless = headless
        self.max_pages = int(max_pages)
        self.rate_limit_seconds = float(rate_limit_seconds)
        self.resolved_url: str | None = None

    def scrape(self) -> list[dict[str, Any]]:
        """Run the full scraping pipeline. Returns list of review dicts."""
        logger.info("Starting scrape for URL: %s", self.google_maps_url)
        with sync_playwright() as p:
            # Launch with anti-detection flags
            browser = p.chromium.launch(
                channel="chrome",  # System Chrome: real GPU/WebGL + genuine TLS fingerprint
                headless=self.headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                ]
            )
            try:
                # Create context with realistic settings
                context = browser.new_context(
                    user_agent=USER_AGENT,
                    viewport={'width': 1920, 'height': 1080},
                    locale='en-US',
                    timezone_id='Asia/Kolkata',
                )
                page = context.new_page()

                # Capture JS console errors for diagnostics
                console_errors = []
                page.on("console", lambda msg: console_errors.append(
                    f"[{msg.type}] {msg.text}"
                ) if msg.type in ("error", "warning") else None)
                page.on("pageerror", lambda err: console_errors.append(
                    f"[PAGE ERROR] {err.message}"
                ))

                # NOTE: playwright-stealth v1.0.6 is NOT used — its init scripts
                # crash Google Maps' JS ("utils is not defined", "opts is not defined").
                # Instead we rely on:
                #   - channel="chrome" → real TLS fingerprint, real chrome.runtime
                #   - --disable-blink-features=AutomationControlled → navigator.webdriver=false
                #   - warm-up navigation flow → behavioral stealth

                self._navigate_to_reviews(page, console_errors)
                self._scroll_all_reviews(page)
                self._expand_all_reviews(page)
                reviews = self._extract_reviews(page)
                logger.info("Extracted %d reviews", len(reviews))
                return reviews
            finally:
                browser.close()

    def _navigate_to_reviews(self, page: Page, console_errors: list = None) -> None:
        """Navigate to the place and open the reviews tab."""
        if self.place_query:
            self._warm_up_navigate(page)
        else:
            self._direct_navigate(page)

        self._click_reviews_tab(page, console_errors)

    def _dismiss_consent(self, page: Page) -> None:
        """Dismiss Google consent/cookie dialogs if present."""
        try:
            for btn_text in ["Accept all", "I agree", "Accept", "Reject all"]:
                consent = page.locator(f'button:has-text("{btn_text}")')
                if consent.count() > 0:
                    consent.first.click(timeout=3000)
                    time.sleep(random.uniform(1, 2))
                    logger.info("Dismissed consent dialog: '%s'", btn_text)
                    return
            # Also try form-based consent (common in some regions)
            form_btn = page.locator('form[action*="consent"] button')
            if form_btn.count() > 0:
                form_btn.first.click(timeout=3000)
                time.sleep(random.uniform(1, 2))
                logger.info("Dismissed form-based consent dialog")
        except Exception:
            pass

    def _warm_up_navigate(self, page: Page) -> None:
        """Warm-up: google.com -> search place -> click Maps result."""
        # Step 1: Visit google.com
        logger.info("Warm-up step 1/3: visiting google.com")
        page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=30000)
        time.sleep(random.uniform(2, 4))
        self._dismiss_consent(page)

        # Step 2: Search for the place on Google (not Maps directly — avoids blank page)
        logger.info("Warm-up step 2/3: searching Google for '%s'", self.place_query)
        search_box = None
        for sel in ['textarea[name="q"]', 'input[name="q"]']:
            loc = page.locator(sel)
            if loc.count() > 0:
                search_box = loc.first
                break

        if not search_box:
            try:
                page.screenshot(path="debug_google_search.png")
            except Exception:
                pass
            raise RuntimeError("Google search box not found — check debug_google_search.png")

        search_box.click(timeout=5000)
        time.sleep(random.uniform(0.5, 1.0))
        page.keyboard.type(self.place_query, delay=random.randint(50, 120))
        time.sleep(random.uniform(1, 2))
        page.keyboard.press("Enter")

        # Wait for search results
        logger.info("Waiting for Google search results...")
        page.wait_for_load_state("domcontentloaded", timeout=15000)
        time.sleep(random.uniform(3, 5))

        # Step 3: Find and click the Maps/place result
        # Google search shows a Maps card or a "View on Maps" link for place queries
        logger.info("Warm-up step 3/3: clicking Maps result")
        clicked = False

        # Try clicking the place card's "Reviews" or the place title on the Maps card
        maps_selectors = [
            # Google search Maps card links
            'a[href*="maps/place"]',
            'a[href*="/maps?"]',
            'a[data-url*="maps"]',
            # Place knowledge panel
            'a[data-ved][href*="ludocid"]',
            # "View all Google reviews" link
            'a:has-text("Google reviews")',
            'a:has-text("reviews")',
        ]

        for selector in maps_selectors:
            try:
                loc = page.locator(selector)
                if loc.count() > 0:
                    loc.first.click()
                    time.sleep(random.uniform(3, 5))
                    clicked = True
                    logger.info("Clicked Maps link with selector: %s", selector)
                    break
            except Exception:
                continue

        if not clicked:
            # Fallback: navigate directly to the Maps URL (original behavior)
            logger.warning("No Maps link found in search results — falling back to direct navigation")
            self._direct_navigate(page)
            return

        # Wait for Maps to load after clicking
        page.wait_for_load_state("domcontentloaded", timeout=30000)
        time.sleep(random.uniform(3, 5))

        # If we're now on Maps, wait for the search box to confirm it rendered
        self._wait_for_maps_ready(page)

        self.resolved_url = page.url
        logger.info("Resolved URL after warm-up: %s", self.resolved_url)

    def _wait_for_maps_ready(self, page: Page) -> None:
        """Wait for Google Maps page to fully render (up to 15 seconds)."""
        for attempt in range(3):
            time.sleep(random.uniform(2, 4))
            self._dismiss_consent(page)

            # Check if we're on a Maps place page by looking for tab buttons
            tabs = page.locator('button[role="tab"]')
            if tabs.count() >= 2:
                logger.info("Maps place page loaded (%d tabs found)", tabs.count())
                return

            # Also check for the place name heading
            heading = page.locator('h1')
            if heading.count() > 0:
                logger.info("Maps page loaded (heading found: %s)", heading.first.text_content()[:50])
                return

            logger.info("Waiting for Maps to render (attempt %d/3)...", attempt + 1)

        logger.warning("Maps page may not have fully rendered — proceeding anyway")

    def _direct_navigate(self, page: Page) -> None:
        """Fallback: direct navigation to the Maps URL (less reliable)."""
        logger.warning("No place_query configured — using direct navigation (may be blocked)")
        try:
            page.goto(self.google_maps_url, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            logger.warning("domcontentloaded failed: %s, trying load...", e)
            page.goto(self.google_maps_url, wait_until="load", timeout=60000)

        page.wait_for_timeout(5000)
        self.resolved_url = page.url
        logger.info("Resolved URL: %s", self.resolved_url)

    def _click_reviews_tab(self, page: Page, console_errors: list = None) -> None:
        """Click the Reviews tab. Raises RuntimeError if not found (trust gate failed)."""
        clicked = False

        # Strategy 1: Text-based
        reviews_tab = page.locator('button:has-text("Reviews")')
        if reviews_tab.count() > 0:
            reviews_tab.first.click()
            page.wait_for_timeout(3000)
            logger.info("Clicked reviews tab (text-based)")
            clicked = True

        # Strategy 2: Role-based
        if not clicked:
            reviews_tab = page.locator('button[role="tab"]').filter(has_text="Reviews")
            if reviews_tab.count() > 0:
                reviews_tab.first.click()
                page.wait_for_timeout(3000)
                logger.info("Clicked reviews tab (role-based)")
                clicked = True

        # Strategy 3: Click rating / review count button (new Maps UI)
        if not clicked:
            try:
                rating_button = page.locator('button[jsaction="pane.rating.moreReviews"]')
                if rating_button.count() > 0:
                    rating_button.first.click()
                    page.wait_for_timeout(4000)
                    logger.info("Clicked rating button to open reviews")
                    clicked = True
            except Exception as e:
                logger.warning("Failed to click rating button: %s", e)

        if not clicked:
            # Dump diagnostics before failing
            try:
                page.screenshot(path="debug_trust_gate_failed.png")
                logger.info("Saved debug screenshot to debug_trust_gate_failed.png")
            except Exception:
                pass

            # Log page HTML length and body content snippet
            try:
                html_len = page.evaluate("document.documentElement.outerHTML.length")
                body_text = page.evaluate("document.body ? document.body.innerText.substring(0, 500) : 'NO BODY'")
                all_buttons = page.evaluate("Array.from(document.querySelectorAll('button')).map(b => b.textContent.trim()).filter(t => t)")
                logger.info("Page HTML length: %d chars", html_len)
                logger.info("Body text (first 500 chars): %s", body_text)
                logger.info("All buttons on page: %s", all_buttons[:20])
            except Exception as e:
                logger.warning("Failed to dump page content: %s", e)

            # Log JS console errors
            if console_errors:
                logger.info("JS console errors (%d):", len(console_errors))
                for err in console_errors[:10]:
                    logger.info("  %s", err)

            raise RuntimeError(
                "Reviews tab not found — Google served reduced UI (trust gate failed). "
                "Check stealth config and warm-up flow."
            )

    def _scroll_all_reviews(self, page: Page) -> None:
        """Scroll the reviews panel until all reviews are loaded."""
        # Try multiple strategies to find the scrollable container
        
        # Strategy 1: Look for scrollable div with reviews
        logger.info("Looking for review scroll container...")
        
        # First check if reviews exist at all
        review_check = page.locator('div[data-review-id]')
        initial_count = review_check.count()
        logger.info("Initial review count visible: %d", initial_count)
        
        if initial_count == 0:
            logger.warning("No reviews found on page! Check if page loaded correctly.")
            # Take screenshot for debugging
            try:
                page.screenshot(path="debug_no_reviews.png")
                logger.info("Saved debug screenshot to debug_no_reviews.png")
            except Exception:
                pass
            return
        
        # Try to find scrollable container - multiple strategies
        scroll_container = None
        
        # Strategy 1: Original specific selectors
        for selector in [
            'div[role="main"] div.m6QErb.DxyBCb.kA9KIf.dS8AEf',
            'div[role="main"] div.m6QErb.DxyBCb.kA9KIf',
            'div[role="main"] div.m6QErb',
        ]:
            try:
                loc = page.locator(selector)
                if loc.count() > 0:
                    scroll_container = loc.first
                    logger.info("Found scroll container with selector: %s", selector)
                    break
            except Exception:
                continue
        
        # Strategy 2: Look for any scrollable div containing reviews
        if not scroll_container:
            logger.info("Trying to find parent container of reviews...")
            try:
                # Get the parent of the first review
                first_review = page.locator('div[data-review-id]').first
                # Try to get scrollable parent
                parent = first_review.locator('xpath=ancestor::div[@role="main"]//div[contains(@class, "m6QErb")]')
                if parent.count() > 0:
                    scroll_container = parent.first
                    logger.info("Found scroll container via review parent")
            except Exception:
                pass
        
        if not scroll_container:
            logger.warning("Could not find specific scroll container, using main role")
            scroll_container = page.locator('div[role="main"]').first

        # Scroll and load reviews
        prev_count = 0
        no_change_iterations = 0
        
        for scroll_num in range(self.max_pages):
            # Count current reviews
            review_elements = page.locator('div[data-review-id]')
            current_count = review_elements.count()

            if current_count == prev_count:
                no_change_iterations += 1
                if no_change_iterations >= 3:
                    logger.info("No new reviews after %d scrolls (total: %d). Done.", scroll_num, current_count)
                    break
            else:
                no_change_iterations = 0

            prev_count = current_count
            
            # Scroll
            try:
                scroll_container.evaluate("el => el.scrollTop = el.scrollHeight")
            except Exception:
                # Fallback: keyboard scroll
                page.keyboard.press("PageDown")
            
            time.sleep(float(self.rate_limit_seconds))
            page.wait_for_timeout(1000)

            if scroll_num % 10 == 0:
                logger.info("Scroll %d, reviews loaded: %d", scroll_num, current_count)

        logger.info("Scrolling complete. Total reviews visible: %d", prev_count)

    def _expand_all_reviews(self, page: Page) -> None:
        """Click all 'More' buttons to expand truncated review text."""
        more_buttons = page.locator('button.w8nwRe.kyuRq')
        count = more_buttons.count()
        if count > 0:
            logger.info("Found %d 'More' buttons, expanding reviews...", count)
            for i in range(count):
                try:
                    more_buttons.nth(i).click(timeout=500)
                    page.wait_for_timeout(100)
                except Exception:
                    pass

    def _extract_reviews(self, page: Page) -> list[dict[str, Any]]:
        """Extract review data from the loaded page."""
        now = datetime.now(IST)
        reviews: list[dict[str, Any]] = []

        review_elements = page.locator('div[data-review-id]')
        count = review_elements.count()
        logger.info("Extracting data from %d review elements", count)

        review_url = self.resolved_url or self.google_maps_url
        seen: set[str] = set()

        for i in range(count):
            try:
                el = review_elements.nth(i)

                # Prefer Google's own stable DOM ID for deduplication
                dom_id = el.get_attribute("data-review-id")
                if dom_id and dom_id in seen:
                    continue

                review = self._parse_single_review(el, now, review_url, dom_id)
                if not review:
                    continue

                dedupe_key = dom_id or review["review_id"]
                if dedupe_key in seen:
                    continue

                seen.add(dedupe_key)
                reviews.append(review)

            except Exception as exc:
                logger.error("Failed to parse review %d: %s", i, exc)

        logger.info("Deduped reviews: %d unique", len(reviews))
        return reviews

    def _parse_single_review(
        self, el: Any, now: datetime, review_url: str, dom_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Parse a single review element into a dict."""
        # Reviewer name
        name_el = el.locator('div.d4r55').first
        reviewer_name = name_el.inner_text() if name_el.count() > 0 else "Unknown"

        # Rating
        rating_el = el.locator('span[role="img"]').first
        rating = 0
        if rating_el.count() > 0:
            aria = rating_el.get_attribute("aria-label") or ""
            match = re.search(r"(\d)\s+star", aria.lower())
            if match:
                rating = int(match.group(1))

        if rating == 0:
            return None

        # Review text
        text_el = el.locator('span.wiI7pd').first
        review_text = text_el.inner_text() if text_el.count() > 0 else None

        # Relative date
        date_el = el.locator('span.rsqaWe').first
        relative_date = date_el.inner_text() if date_el.count() > 0 else ""
        review_date = parse_relative_date(relative_date, now)

        # Use Google's stable DOM ID when available, fall back to SHA256 hash
        review_id = dom_id or generate_review_id(reviewer_name, review_date, review_text)

        return {
            "review_id": review_id,
            "reviewer_name": reviewer_name,
            "rating": rating,
            "review_text": review_text,
            "review_date": review_date,
            "review_link": review_url,
        }