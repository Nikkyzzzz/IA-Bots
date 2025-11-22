# ============================== b6.py — Banking Processing ==============================
import os, time, random
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import base64
from pathlib import Path
from io import BytesIO

import blogic     # adapter for banking + runners
import blogic6    # bot functions + PROCESS_TITLES

LEFT_LOGO_PATH = "logo.png"

# ---------------- header helpers ----------------
def _data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists(): return ""
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in (".jpg",".jpeg") else "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _scroll_to_brand_if_needed(s):
    if s.pop("_force_scroll_top", False):
        components.html(
            """
            <script>
              (function(){
                function go(){
                  try{
                    const doc = window.parent && window.parent.document ? window.parent.document : document;
                    const el = doc.getElementById("__ab_header__");
                    if (el && el.scrollIntoView){
                      el.scrollIntoView({block:"start", inline:"nearest"});
                    } else { (window.parent||window).scrollTo(0,0); }
                  }catch(e){ (window.parent||window).scrollTo(0,0); }
                }
                go(); setTimeout(go,50); setTimeout(go,150); setTimeout(go,300); setTimeout(go,600);
              })();
            </script>
            """,
            height=0,
        )

def _render_header():
    st.markdown(
        """
        <style>
          .ab-wrap * { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important; }
          .ab-title {
            font-weight:860; letter-spacing:.02em;
            background: linear-gradient(90deg, #1e3a8a 0%, #60a5fa 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; color: transparent;
            font-size: clamp(30px, 3.2vw, 38px); line-height: 1.05;
          }
          .ab-underline { height:8px; border-radius:9999px; background:#ff7636; margin-top:8px; width:320px; max-width:30vw; }
          .ab-subtitle { margin-top:10px; font-weight:700; color:#0f172a; font-size: clamp(16px, 1.6vw, 20px); }
          .ab-spacer { height: 1.2em; }
          .hdr-grid { display:grid; grid-template-columns:268px 1fr; align-items:center; column-gap:16px; margin-bottom:8px; }
          .hdr-brand { display:flex; justify-content:flex-end; align-items:center; height:168px; position:relative; }
          #__ab_header__ { scroll-margin-top: 96px; }
          @media (max-width: 640px){ #__ab_header__{ scroll-margin-top: 120px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-wrap">', unsafe_allow_html=True)

    logo_src = _data_uri(LEFT_LOGO_PATH)
    st.markdown(
        f"""
        <div id="__ab_header__">
          <div class="hdr-grid">
            <div class="hdr-logo" style="width:268px;">
              {'<img src="' + logo_src + '" width="154" height="100" alt="Logo" style="display:block;" />' if logo_src else ''}
            </div>
            <div class="hdr-brand" style="text-align:right;">
              <div>
                <div class="ab-title">Audit&nbsp;Bots</div>
                <div class="ab-underline" style="margin-left:auto;"></div>
                <div class="ab-subtitle">Banking • Processing</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)

# ---------------- navigation ----------------
def _final_nav():
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("⟵ Back to Selection", key="b6_back_after"):
            st.session_state["_force_scroll_top"] = True
            st.session_state.page = "bank5"
            st.rerun()
    with col2:
        if st.button("View Output ➜", key="b6_view_output"):
            st.session_state["_force_scroll_top"] = True
            st.session_state.page = "bank7"
            st.rerun()

# ---------------- main ----------------
def render_bank6():
    s = st.session_state
    st.set_page_config(layout="wide")
    _render_header(); _scroll_to_brand_if_needed(s)

    # guard
    if not s.get("sheet_mapping_pairs") or not s.get("column_mapping_pairs"):
        st.warning("No mappings found. Please Completed sheet & field mapping first.")
        if st.button("⟵ Back", key="b6_back_no_map"):
            s.page = "bank5"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True); return

    if "processing_started" not in s: s.processing_started = False
    if "processing_done" not in s: s.processing_done = False
    if "proc_status" not in s: s.proc_status = {code: "Pending" for code in s.selected_bots}
    if "results" not in s: s.results = {}

    sel_codes = list(s.get("selected_bots", []) or [])
    if not sel_codes:
        st.warning("No bots selected. Please go back and select at least one bot.")
        if st.button("⟵ Back", key="b6_back_no_bots"):
            s.page = "bank5"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True); return

    # Initialize bot progress tracking
    if "bot_progress" not in s: s.bot_progress = {}
    if "bot_processing_order" not in s: s.bot_processing_order = []
    if "current_processing_bot_index" not in s: s.current_processing_bot_index = 0
    if "last_bot_update_time" not in s: s.last_bot_update_time = 0

    # render bot rows
    ui_refs = {}
    # Mapping of old bot names to new display names
    BOT_DISPLAY_NAMES = {
        "zero_or_null_roi_loans": "Loans with Zero/Null ROI",
        "standard_accounts_with_uri_zero": "Standard Accounts having Zero URI",
        "provision_verification_substandard_npa": "Provision Status – Sub-Standard NPA",
        "restructured_standard_accounts": "Restructured Standard Accounts",
        "provision_verification_doubtful3_npa": "Provision Status – Doubtful-3 NPA",
        "npa_fb_accounts_overdue": "NPA FB Accounts with Overdues",
        "negative_amt_outstanding": "Accounts with Negative Outstanding",
        "standard_accounts_overdue_details": "Overdue Standard Accounts",
        "standard_accounts_with_odd_interest": "Standard Accounts with Irregular Interest",
        "agri0_sector_over_limit": "Agriculture Sector – Limit Breach Cases",
        "misaligned_scheme_for_facilities": "Facilities with Scheme Mismatch",
        "Loans & Advances to Blacklisted Areas": "Loans in Blacklisted Areas",
        "Blank Asset Classification": "Accounts with Blank Asset Classification"
    }

    for code in sel_codes:
        # Use new display name if available
        _, orig_bot_name = blogic6.PROCESS_TITLES[code]
        bot_name = BOT_DISPLAY_NAMES.get(code, orig_bot_name)
        row = st.container()
        with row:
            c1, c2 = st.columns([3, 2])
            status_ph = c1.empty()
            # Initialize progress for this bot if not exists
            if code not in s.bot_progress:
                s.bot_progress[code] = {"progress": 0, "status": "Pending"}
            curr_status = s.bot_progress[code]["status"]
            curr_progress = s.bot_progress[code]["progress"]
            # Create progress bar with current progress
            prog_ph = c2.progress(curr_progress)
            sym = {"Pending":"⏳","Processing":"🔄","Completed":"✅","Failed":"❌"}.get(curr_status,"⏳")
            status_ph.markdown(f"- {sym} **{bot_name}** — {curr_status}")
            ui_refs[code] = {"status": status_ph, "prog": prog_ph, "current_status": curr_status}
    
    st.markdown("---")

    # nav before start
    if not s.processing_started and not s.processing_done:
        nav = st.columns([1,1,6])
        if nav[0].button("⟵ Back", key="b6_back_before"):
            s.page="bank5"; st.rerun()
        if nav[1].button("Process ➜", key="b6_start"):
            s.processing_started=True; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True); return

    if s.processing_done:
        _final_nav(); st.markdown('</div>', unsafe_allow_html=True); return

    # ========== Processing ========== #
    file_bytes_map = {
        "Banking": s.get("u_ccis_bytes"),
        "Blacklisted PIN CODE": s.get("u_blacklist_bytes"),
        "Loan Book (31.03.2025)": s.get("u_loan_mar_bytes"),
        "Loan Book (30.06.2025)": s.get("u_loan_jun_bytes"),
    }

    # Animated Bot Processing System
    def animate_bot_progress():
        """Animate progress bars for each bot with random increments and timing"""
        # Set up processing order if not already done
        if not s.bot_processing_order:
            s.bot_processing_order = sel_codes.copy()
            s.current_processing_bot_index = 0
            s.last_bot_update_time = time.time()
            s.bot_timings = {}  # Track per-bot total duration + start time

        current_time = time.time()

        if s.current_processing_bot_index < len(s.bot_processing_order):
            current_bot = s.bot_processing_order[s.current_processing_bot_index]
            bot_data = s.bot_progress[current_bot]

            if bot_data["status"] == "Pending":
                # Start this bot
                bot_data["status"] = "Processing"
                bot_data["progress"] = 1
                _, bot_name = blogic6.PROCESS_TITLES[current_bot]
                ui_refs[current_bot]["status"].markdown(f"- 🔄 **{bot_name}** — Processing")
                ui_refs[current_bot]["prog"].progress(1)

                # Assign random total duration
                total_time = random.uniform(2.3, 5.0)
                s.bot_timings[current_bot] = {
                    "start": current_time,
                    "duration": total_time
                }

            elif bot_data["status"] == "Processing":
                elapsed = current_time - s.bot_timings[current_bot]["start"]
                total_time = s.bot_timings[current_bot]["duration"]

                # Calculate expected progress
                target_progress = min(100, int((elapsed / total_time) * 100))

                # Ensure strictly increasing progress
                if target_progress > bot_data["progress"]:
                    bot_data["progress"] = target_progress
                    ui_refs[current_bot]["prog"].progress(bot_data["progress"])

                # If finished
                if bot_data["progress"] >= 100:
                    bot_data["status"] = "Completed"
                    _, bot_name = blogic6.PROCESS_TITLES[current_bot]
                    ui_refs[current_bot]["status"].markdown(f"- ✅ **{bot_name}** — Completed")
                    ui_refs[current_bot]["prog"].progress(100)

                    # Add small random delay before next bot
                    delay = random.uniform(0.5, 1.4)
                    s.next_bot_start_time = current_time + delay
                    s.current_processing_bot_index += 1

        # Check if all bots are Completed
        all_done = all(s.bot_progress[code]["status"] == "Completed" for code in sel_codes)
        return all_done


    # Check if we should start next bot
    current_time = time.time()
    if (hasattr(s, 'next_bot_start_time') and 
        current_time >= s.next_bot_start_time and 
        s.current_processing_bot_index < len(s.bot_processing_order)):
        # Time to start next bot - clear the delay
        delattr(s, 'next_bot_start_time')

    # Animate the progress
    all_bots_Completed = animate_bot_progress()
    
    # Only refresh if bots are still processing AND significant time has passed
    if not all_bots_Completed:
        # Only refresh every 1 second instead of 0.5 to reduce overhead
        time.sleep(1.0)
        st.rerun()
    else:
        # All bots Completed - run actual processing to get real results
        if not s.get("real_processing_done", False):
            with st.spinner("Finalizing results..."):
                results, proc_status, raw_dfs = blogic.run_all_bots_with_mappings(
                    file_bytes_map=file_bytes_map,
                    sheet_mapping_pairs=s.sheet_mapping_pairs,
                    column_mapping_pairs=s.column_mapping_pairs,
                )
                
                # Save results
                s.results = results
                s.proc_status.update(proc_status)
                s.raw_dfs = raw_dfs
                s.real_processing_done = True

    # Final UI updates only after real processing is Completed
    if s.get("real_processing_done", False):
        # update UI after run - ensure all bots show as Completed
        for code in sel_codes:
            if code in ui_refs:
                status = s.proc_status.get(code, "Completed")  # Default to Completed since animation finished
                _, orig_bot_name = blogic6.PROCESS_TITLES[code]
                bot_name = BOT_DISPLAY_NAMES.get(code, orig_bot_name)
                status_ph = ui_refs[code]["status"]; prog_ph = ui_refs[code]["prog"]
                sym = {"Pending":"⏳","Completed":"✅","Failed":"❌"}.get(status,"✅")
                status_ph.markdown(f"- {sym} **{bot_name}** — Completed")
                prog_ph.progress(100)
                s.bot_progress[code]["status"] = "Completed"
                s.bot_progress[code]["progress"] = 100
        
        # Mark processing as done
        s.processing_done = True
    
    # Show final navigation only when everything is Completed
    if s.get("real_processing_done", False):
        _final_nav()
    st.markdown('</div>', unsafe_allow_html=True)
