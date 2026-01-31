"""
Streamlit UI for the Local Reputation Agent.

Two tabs:
  - Pulse: reputation score, sentiment breakdown, top complaints
  - Review Queue: human-in-the-loop approval for AI-drafted responses

Usage:
    streamlit run agent/app.py
"""

import json
import os
import sys

# Add agent directory to path for imports
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AGENT_DIR)

import streamlit as st

from db import init_db, get_all_reviews, get_complaint_counts, get_pending_responses, approve_response, reject_response
from agent_config import BUSINESS_NAME

init_db()

st.set_page_config(page_title=f"{BUSINESS_NAME} — Reputation Agent", layout="wide")
st.title(f"{BUSINESS_NAME} — Reputation Agent")

tab_pulse, tab_queue = st.tabs(["Pulse", "Review Queue"])

# =============================================================================
# TAB 1: PULSE
# =============================================================================
with tab_pulse:
    reviews = get_all_reviews()

    if not reviews:
        st.info("No reviews processed yet. Run the scheduler first: `python agent/scheduler.py --once`")
    else:
        # --- Reputation Score ---
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        total_weight = 0
        weighted_sum = 0
        for r in reviews:
            weight = 1
            try:
                review_dt = datetime.fromisoformat(r["review_date"])
                if review_dt.tzinfo is None:
                    review_dt = review_dt.replace(tzinfo=timezone.utc)
                if (now - review_dt).days <= 30:
                    weight = 2
            except (ValueError, TypeError):
                pass
            weighted_sum += r["rating"] * weight
            total_weight += weight

        reputation_score = weighted_sum / total_weight if total_weight else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Reputation Score", f"{reputation_score:.1f} / 5.0")
        col2.metric("Total Reviews", len(reviews))

        # Sentiment breakdown
        pos = sum(1 for r in reviews if r["overall_sentiment"] == "Positive")
        neu = sum(1 for r in reviews if r["overall_sentiment"] == "Neutral")
        neg = sum(1 for r in reviews if r["overall_sentiment"] == "Negative")
        col3.metric("Positive", pos)
        col4.metric("Negative", neg)

        # --- Top 3 Complaints ---
        st.subheader("Top Complaints")
        complaints = get_complaint_counts()
        if complaints:
            for aspect, count in complaints[:3]:
                st.write(f"**{aspect.capitalize()}** — {count} negative mentions")
        else:
            st.write("No complaints detected yet.")

        # --- Recent Negative Reviews ---
        st.subheader("Recent Negative Reviews")
        negative = [r for r in reviews if r["overall_sentiment"] == "Negative"][:5]
        for r in negative:
            urgent_tag = " | URGENT" if r["urgent"] else ""
            st.markdown(
                f"**{r['reviewer_name']}** ({r['rating']}/5{urgent_tag}) — "
                f"_{r['review_text'][:120]}{'...' if len(r['review_text']) > 120 else ''}_"
            )

# =============================================================================
# TAB 2: REVIEW QUEUE
# =============================================================================
with tab_queue:
    pending = get_pending_responses()

    if not pending:
        st.info("No pending reviews in the queue. The agent will add drafts when negative reviews are detected.")
    else:
        st.write(f"**{len(pending)} review(s) pending approval**")

        for item in pending:
            with st.container(border=True):
                st.markdown(
                    f"**{item['reviewer_name']}** — "
                    f"{'⭐' * item['rating']} ({item['rating']}/5)"
                    f"{' | URGENT' if item['urgent'] else ''}"
                )
                st.markdown(f"> {item['review_text']}")

                # Editable draft
                edited = st.text_area(
                    "AI Draft Response (edit before approving):",
                    value=item["draft_response"],
                    key=f"draft_{item['id']}",
                    height=120,
                )

                col_approve, col_reject, _ = st.columns([1, 1, 4])
                with col_approve:
                    if st.button("Approve", key=f"approve_{item['id']}", type="primary"):
                        approve_response(item["id"], edited)
                        st.success("Approved!")
                        st.rerun()
                with col_reject:
                    if st.button("Reject", key=f"reject_{item['id']}"):
                        reject_response(item["id"])
                        st.warning("Rejected")
                        st.rerun()
