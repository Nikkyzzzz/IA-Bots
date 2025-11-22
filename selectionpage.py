# ============================== selectionpage.py — Choose Bots ==============================
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import base64

import logic6  # uses PROCESS_TITLES, BOT_NAMES
# categories used everywhere in your app
CATEGORIES_ORDER = ("P2P", "O2C", "H2R")
LEFT_LOGO_PATH = "logo.png"

# ---------- header helpers (match look & feel on other pages) ----------
def _data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    else:
        mime = "image/png"
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
                  if (el && el.scrollIntoView) {
                    el.scrollIntoView({block:"start", inline:"nearest"});
                  } else {
                    (window.parent||window).scrollTo(0,0);
                  }
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
          .hint { color:#475569; font-size: 0.92rem; margin: .25rem 0 .75rem 0; }
          .box { border:1px solid rgba(15,23,42,.06); border-radius:14px; padding:12px; background:#fff; }
          .split { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
          @media (max-width: 980px){ .split { grid-template-columns:1fr; } }
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
                <div class="ab-subtitle">Select Bots to Process</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)

# ---------- helpers ----------
def _available_categories():
    s = st.session_state
    return [c for c in CATEGORIES_ORDER if c in (s.sheet_mapping_pairs or {})]

def _codes_for_category(category: str):
    return [code for code, (cat, _) in logic6.PROCESS_TITLES.items() if cat == category]

def _default_codes(cats_present):
    codes = []
    for c in cats_present:
        codes.extend(_codes_for_category(c))
    return codes

# ---------- page ----------
def render_selectionpage():
    s = st.session_state
    st.set_page_config(layout="wide")

    _render_header()
    _scroll_to_brand_if_needed(s)

    # guard: need mappings already done (comes after Review)
    if not s.get("sheet_mapping_pairs") or not s.get("column_mapping_pairs"):
        st.warning("No mappings found. Please complete sheet & field mapping first.")
        if st.button("⟵ Back to Review"):
            s["_force_scroll_top"] = True
            s.page = "fourth"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    cats = _available_categories()
    if not cats:
        st.warning("No categories available. Map at least one category to proceed.")
        if st.button("⟵ Back to Review", key="sel_back_no_cats"):
            s["_force_scroll_top"] = True
            s.page = "fourth"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ------- Build code lists -------
    codes_by_cat = {cat: _codes_for_category(cat) for cat in cats}
    valid_order = [c for cat in cats for c in codes_by_cat[cat]]  # ordered, unique

    # ------- Bootstrap selection state (start empty) -------
    if "selected_bots" not in s or not isinstance(s.selected_bots, list):
        s.selected_bots = []  # ✅ no pre-selection

    # Initialize per-bot checkbox state keys (default False)
    for code in valid_order:
        key = f"sel_{code}"
        if key not in s:
            s[key] = (code in s.selected_bots)

    # Helper to check if all codes are selected
    def _all_selected(codes):
        return all(s.get(f"sel_{c}", False) for c in codes) if codes else False

    # ---------- CALLBACKS ----------
    def _toggle_all_bots():
        val = bool(st.session_state.get("sel_all_bots", False))
        for c in valid_order:
            st.session_state[f"sel_{c}"] = val

    def _make_toggle_cat(cat, cat_codes):
        def _cb():
            key = f"sel_all_{cat}"
            val = bool(st.session_state.get(key, False))
            for c in cat_codes:
                st.session_state[f"sel_{c}"] = val
        return _cb

    # st.markdown('<div class="hint">Use the checkboxes below. “Select all” instantly syncs the list—no double clicks.</div>', unsafe_allow_html=True)

    # ================= GLOBAL SELECT-ALL (top, like before) =================
    with st.container():
        st.markdown('<div class="box">', unsafe_allow_html=True)
        st.markdown('')
        # Derive from current bot states (set BEFORE rendering the widget)
        st.session_state["sel_all_bots"] = _all_selected(valid_order)
        st.checkbox(
            "Select all bots",
            key="sel_all_bots",
            on_change=_toggle_all_bots,
            help="Toggle all bots across all categories"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= Two-column grouped picker with per-category select-all (at top) =================
    with st.container():
        st.markdown('<div class="split">', unsafe_allow_html=True)
        colA, colB = st.columns(2)
        side = 0

        for cat in cats:
            codes = codes_by_cat[cat]
            tgt = colA if side == 0 else colB
            with tgt:
                st.markdown(f"#### {cat} Bots")
                with st.container():
                    st.markdown('<div class="box">', unsafe_allow_html=True)

                    # ----- Category select-all (TOP, like before) -----
                    cat_key = f"sel_all_{cat}"
                    st.session_state[cat_key] = _all_selected(codes)  # set BEFORE rendering category widget
                    st.checkbox(
                        f"Select all in {cat}",
                        key=cat_key,
                        on_change=_make_toggle_cat(cat, codes),
                        help="Toggle all bots in this category"
                    )

                    # ----- Individual bot checkboxes -----
                    for code in codes:
                        _, bot_name = logic6.PROCESS_TITLES[code]
                        st.checkbox(bot_name, key=f"sel_{code}")

                    st.markdown('</div>', unsafe_allow_html=True)
            side ^= 1

        st.markdown('</div>', unsafe_allow_html=True)

    # =============== Derive selected_bots ===============
    s.selected_bots = [c for c in valid_order if s.get(f"sel_{c}", False)]
    s.selected_categories = sorted({logic6.PROCESS_TITLES[c][0] for c in s.selected_bots})

    # ---------- Navigation ----------
    nav = st.columns([1, 1, 6])
    with nav[0]:
        if st.button("⟵ Back", key="sel_back"):
            s["_force_scroll_top"] = True
            s.page = "fourth"; st.rerun()
    with nav[1]:
        if st.button("Continue ➜", key="sel_next"):
            s["_force_scroll_top"] = True
            # Deduplicate & keep only valid/ordered codes
            s.selected_bots = [c for c in valid_order if c in set(s.selected_bots)]
            s.selected_categories = sorted({logic6.PROCESS_TITLES[c][0] for c in s.selected_bots})
            s.page = "processpage"; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
