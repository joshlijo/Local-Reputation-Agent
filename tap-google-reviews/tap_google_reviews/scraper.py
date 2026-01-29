"""Playwright-based Google Maps reviews scraper with robust selectors."""

from __future__ import annotations

import hashlib
import logging
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


class GoogleReviewsScraper:
    """Scrapes Google Maps reviews using Playwright."""

    def __init__(
        self,
        google_maps_url: str,
        headless: bool = True,
        max_pages: int = 100,
        rate_limit_seconds: float = 1.0,
    ) -> None:
        self.google_maps_url = google_maps_url
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
                headless=self.headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-web-security',
                ]
            )
            try:
                # Create context with realistic settings
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080},
                    locale='en-US',
                    timezone_id='Asia/Kolkata',
                )
                page = context.new_page()
                
                # Hide automation markers
                page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => false});
                    window.navigator.chrome = {runtime: {}};
                """)
                
                self._navigate_to_reviews(page)
                self._scroll_all_reviews(page)
                self._expand_all_reviews(page)
                reviews = self._extract_reviews(page)
                logger.info("Extracted %d reviews", len(reviews))
                return reviews
            finally:
                browser.close()

    def _navigate_to_reviews(self, page: Page) -> None:
        """Navigate to the Google Maps page and open the reviews tab."""
        logger.info("Navigating to %s", self.google_maps_url)
        
        # Use domcontentloaded instead of networkidle (more reliable)
        try:
            page.goto(self.google_maps_url, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            logger.warning("domcontentloaded failed: %s, trying load...", e)
            page.goto(self.google_maps_url, wait_until="load", timeout=60000)
        
        # Wait for page to settle
        page.wait_for_timeout(5000)
        
        self.resolved_url = page.url
        logger.info("Resolved URL: %s", self.resolved_url)

        # Try to click reviews tab - multiple strategies
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