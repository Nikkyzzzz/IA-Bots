# ============================== processpage.py — Processing ==============================
import os, re, importlib, time, random, numbers
from pathlib import Path
from io import BytesIO

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import base64

import logic   # adapter that uses mapped fields and returns canonical DFs
import logic6  # existing logic: PROCESS_TITLES, bots, helpers

LEFT_LOGO_PATH = "logo.png"
CATEGORIES_ORDER = ("P2P", "O2C", "H2R")

# -------------------------- Navigation helpers -------------------------- #
REVIEW_PAGE_CANDIDATES = [
    "pages/4_Review.py",
    "pages/04_Review.py",
    "pages/Review.py",
]

def _safe_switch_page(candidates, fallback_state_key):
    for target in candidates:
        try:
            st.switch_page(target)
            return
        except Exception:
            continue
    st.session_state.page = fallback_state_key
    st.rerun()
# ------------------------------------------------------------------------ #

# ---------- header helpers ----------
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
                <div class="ab-subtitle">Processing</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)

# ---------- simple helpers ----------
def _codes_for_category(category: str):
    return [code for code, (cat, _) in logic6.PROCESS_TITLES.items() if cat == category]

def _available_categories():
    s = st.session_state
    return [c for c in CATEGORIES_ORDER if c in (s.sheet_mapping_pairs or {})]

# ======================================================================
#            ⚙️ Parameter Editor — persistent to logic6.py
#            (ported in from the other processpage)
# ======================================================================

# Map UI keys → variable names in logic6.py (avoid colliding with function names)
VAR_NAME_MAP = {
    "variable1": "PO_GRN_Invoice",          # amount mismatch threshold for P2P2
    "variable2": "Generate_self_approved",  # toggle/limit
    "variable3": "vendor_year_threshold",   # months
    "variable4": "vendor_daily_threshold",  # count/day
    "variable5": "OVERDUE_DAYS_THRESHOLD",  # do NOT shadow function name
}

# Defaults if not present in logic6.py
DEFAULTS = {
    "PO_GRN_Invoice": 1000.0,
    "Generate_self_approved": 1,
    "vendor_year_threshold": 12,
    "vendor_daily_threshold": 5,
    "OVERDUE_DAYS_THRESHOLD": 7,
}

def _as_number(val, default):
    """Return a float value; if val is callable/invalid, fall back to default."""
    if callable(val):
        return float(default)
    if isinstance(val, numbers.Number):
        return float(val)
    try:
        return float(val)
    except Exception:
        return float(default)

def _get_logic6_value(name):
    default = DEFAULTS.get(name, 0)
    try:
        v = getattr(logic6, name)
    except Exception:
        return default
    return _as_number(v, default)

def _read_current_params_from_logic6():
    # returns UI-keyed dict, each value numeric
    return { ui_key: _get_logic6_value(py_name) for ui_key, py_name in VAR_NAME_MAP.items() }

def _write_params_into_logic6(new_vals: dict):
    """
    Persist numeric values into logic6.py.
    - Uses VAR_NAME_MAP to write only to safe variable names.
    - Skips writing if the target name in logic6 is a callable (function).
    """
    logic6_path = os.path.join(os.path.dirname(__file__), "logic6.py")
    try:
        with open(logic6_path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        st.error(f"Could not read logic6.py: {e}")
        return False

    def _has_assignment(name: str) -> bool:
        pat = rf"(^\s*{re.escape(name)}\s*=\s*)(.+)$"
        return bool(re.search(pat, src, flags=re.MULTILINE))

    def _replace_assignment(name: str, value_repr: str) -> str:
        # Replace:  ^\s*<name>\s*=\s*<anything to end-of-line>
        pat = rf"(^\s*{re.escape(name)}\s*=\s*)(.+)$"
        # Use a callback so we can concatenate the kept prefix with the literal value
        return re.sub(pat, lambda m: m.group(1) + value_repr, src, flags=re.MULTILINE)

    changed = False
    for ui_key, py_name in VAR_NAME_MAP.items():
        val = new_vals.get(ui_key, DEFAULTS[py_name])

        # Never overwrite a function by mistake
        try:
            existing = getattr(logic6, py_name)
            if callable(existing):
                st.warning(f"Skipped saving '{py_name}' because it is a function in logic6.py.")
                continue
        except Exception:
            pass

        # nice numeric literal (no trailing .0 for ints)
        try:
            fval = float(val)
            val_repr = str(int(fval)) if fval.is_integer() else str(fval)
        except Exception:
            fdef = float(DEFAULTS[py_name])
            val_repr = str(int(fdef)) if fdef.is_integer() else str(fdef)

        if _has_assignment(py_name):
            new_src = _replace_assignment(py_name, val_repr)
            if new_src != src:
                src = new_src
                changed = True
        else:
            src = src.rstrip() + f"\n{py_name} = {val_repr}\n"
            changed = True

    if changed:
        try:
            with open(logic6_path, "w", encoding="utf-8") as f:
                f.write(src)
        except Exception as e:
            st.error(f"Could not write logic6.py: {e}")
            return False

        try:
            importlib.reload(logic6)
        except Exception as e:
            st.warning(f"Saved, but reload failed: {e}")

    return True

def _render_param_editor():
    st.markdown("### ⚙️ Edit Parameters")
    st.caption("These values are saved **permanently** to `logic6.py` and applied immediately.")

    current = _read_current_params_from_logic6()

    c1, c2 = st.columns(2)
    with c1:
        v1 = st.number_input(
            "P2P : PO GRN Invoice Mismatch — PO_GRN_Invoice (Amount threshold)",
            value=_as_number(current.get("variable1", DEFAULTS["PO_GRN_Invoice"]), DEFAULTS["PO_GRN_Invoice"]),
            step=100.0, min_value=0.0
        )
        v2 = st.number_input(
            "P2P : Split Order — Generate_self_approved (Amount threshold)",
            value=_as_number(current.get("variable2", DEFAULTS["Generate_self_approved"]), DEFAULTS["Generate_self_approved"]),
            step=1.0, min_value=0.0
        )
        v3 = st.number_input(
            "P2P : Duplicte Vendors — vendor_year_threshold (Amount threshold)",
            value=_as_number(current.get("variable3", DEFAULTS["vendor_year_threshold"]), DEFAULTS["vendor_year_threshold"]),
            step=1.0, min_value=0.0
        )
    with c2:
        v4 = st.number_input(
            "P2P : Duplicate Vendors — vendor_daily_threshold (Amount threshold)",
            value=_as_number(current.get("variable4", DEFAULTS["vendor_daily_threshold"]), DEFAULTS["vendor_daily_threshold"]),
            step=1.0, min_value=0.0
        )
        v5 = st.number_input(
            "O2C : Overdue Delivery — check_overdue_delivery (Days threshold)",
            value=_as_number(current.get("variable5", DEFAULTS["OVERDUE_DAYS_THRESHOLD"]), DEFAULTS["OVERDUE_DAYS_THRESHOLD"]),
            step=1.0, min_value=0.0
        )
    # ---- Buttons row (aligned side by side) ----
    b1, b2, _sp = st.columns([1, 1, 6])
    with b1:
        back = st.button("⟵ Back", use_container_width=True, key="params_back")
    with b2:
        save = st.button("💾 Save", type="primary", use_container_width=True, key="params_save")



    if save:
        payload = {
            "variable1": v1,
            "variable2": v2,
            "variable3": v3,
            "variable4": v4,
            "variable5": v5,  # writes to OVERDUE_DAYS_THRESHOLD
        }
        if _write_params_into_logic6(payload):
            st.success("Saved to logic6.py and reloaded.", icon="✅")
            st.session_state.show_param_editor = False
            st.rerun()

    if back and not save:
        st.session_state.show_param_editor = False
        st.rerun()

# ======================================================================
#                        Processing helpers
# ======================================================================

def _final_nav():
    left, right = st.columns([1, 4])
    with left:
        if st.button("⟵ Back to Selection", key="pp_back_after"):
            st.session_state["_force_scroll_top"] = True
            st.session_state.page = "selectionpage"; st.rerun()
    with right:
        if st.button("View Output ➜", key="pp_view_output"):
            st.session_state.page = "fifth"
            st.session_state["_force_scroll_top"] = True
            st.rerun()

# ---------- main page ----------
def render_processpage():
    s = st.session_state
    st.set_page_config(layout="wide")

    # Header
    _render_header()
    _scroll_to_brand_if_needed(s)

    # Guards
    if not s.get("sheet_mapping_pairs") or not s.get("column_mapping_pairs"):
        st.warning("No mappings found. Please complete sheet & field mapping first.")
        if st.button("⟵ Back", key="pp_back_no_mappings"):
            s["_force_scroll_top"] = True
            _safe_switch_page(REVIEW_PAGE_CANDIDATES, fallback_state_key="fourth")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Compute categories to show:
    all_mapped_cats = _available_categories()
    sel_codes = list(s.get("selected_bots", []) or [])
    if sel_codes:
        sel_cats = sorted({logic6.PROCESS_TITLES[c][0] for c in sel_codes if c in logic6.PROCESS_TITLES})
        cats = [c for c in CATEGORIES_ORDER if (c in all_mapped_cats and c in sel_cats)]
    else:
        cats = all_mapped_cats  # no prior selection → show all mapped categories

    if not cats:
        st.warning("No categories selected. Please go back and select at least one bot.")
        if st.button("⟵ Back", key="pp_back_no_cats"):
            s["_force_scroll_top"] = True
            s.page = "selectionpage"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ---------------- Init processing state ---------------- #
    if "processing_started" not in s: s.processing_started = False
    if "processing_done" not in s: s.processing_done = False
    if "proc_status" not in s:
        s.proc_status = {code: "Pending" for code in logic6.PROCESS_TITLES}
    if "results" not in s: s.results = {}

    # ---------------- Top action: Threshold editor (minimal addition) ---------------- #
    with st.container():
        if st.button("⚙️ Edit Parameters", key="pp_edit_params"):
            s.show_param_editor = True

    if s.get("show_param_editor", False):
        _render_param_editor()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ---------------- Render selected bots only ---------------- #
    sel_set = set(sel_codes)
    ui_refs: dict[str, dict] = {}
    for cat in cats:
        st.header(f"{cat} Bots")
        any_in_cat = False
        for code in _codes_for_category(cat):
            if sel_set and code not in sel_set:
                continue
            any_in_cat = True
            _, bot_name = logic6.PROCESS_TITLES[code]
            row = st.container()
            with row:
                c1, c2 = st.columns([3, 2])
                with c1:
                    status_ph = st.empty()
                with c2:
                    prog_ph = st.progress(0)

                curr_status = s.proc_status.get(code, "Pending")
                sym = {"Pending":"⏳","Complete":"✅","Failed":"❌"}.get(curr_status, "⏳")
                status_ph.markdown(f"- {sym} **{bot_name}** — {curr_status}")

                if curr_status in ("Complete", "Failed"):
                    prog_ph.progress(100)

                ui_refs[code] = {"status": status_ph, "prog": prog_ph}
        if not any_in_cat:
            st.info("No bots selected in this category.")
        st.markdown("---")

    # ---------- Idle state ----------
    if not s.processing_started and not s.processing_done:
        nav = st.columns([1, 1, 6])
        with nav[0]:
            if st.button("⟵ Back", key="pp_back_before"):
                s["_force_scroll_top"] = True
                s.page = "selectionpage"; st.rerun()
        with nav[1]:
            if st.button("Process ➜", key="pp_start"):
                s.processing_started = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ---------- Already finished ----------
    if s.processing_done:
        _final_nav()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # =============== PROCESS NOW (after Start pressed) ===============
    file_bytes_map = {
        "MASTER": s.get("u_master_bytes"),
        "P2P":    s.get("u_p2p_bytes"),
        "O2C":    s.get("u_o2c_bytes"),
        "H2R":    s.get("u_h2r_bytes"),
    }

    with st.spinner("Preparing dataframes from your mapped sheets & columns..."):
        df_vendor, df_p2p, df_emp_p2p, df_o2c, df_cust, df_emp_h2r, df_att = logic.prepare_dataframes(
            file_bytes_map=file_bytes_map,
            sheet_mapping_pairs=s.sheet_mapping_pairs,
            column_mapping_pairs=s.column_mapping_pairs,
        )

    # expose raw DFs for 5th page
    s.raw_dfs = {
        "VENDOR_RAW":  df_vendor,
        "P2P_RAW":     df_p2p,
        "EMP_P2P_RAW": df_emp_p2p,  # for P2P2 analytics
        "O2C_RAW":     df_o2c,
        "CUST_RAW":    df_cust,
        "EMP_H2R_RAW": df_emp_h2r,
        "ATT_RAW":     df_att,
        # Backward-compatible alias some code might expect:
        "EMP_RAW":     df_emp_p2p,
    }

    if df_p2p is not None:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            df_p2p.to_excel(writer, sheet_name="P2P_Sample (Bots 1-20)", index=False)
        bio.seek(0)
        s.file_bytes = bio.getvalue()
    else:
        s.file_bytes = None

    def _run_bot_and_update(code: str):
        _, bot_name = logic6.PROCESS_TITLES[code]
        status_ph = ui_refs[code]["status"]
        prog_ph   = ui_refs[code]["prog"]

        # --- Smooth pseudo-progress ---
        # Total per-bot delay: random 3–6 seconds.
        # Each increment occurs after a random 1–3s sleep.
        # Progress increments are random and strictly increasing (monotonic).
        total_target = random.uniform(3.0, 6.0)
        elapsed = 0.0
        progress_val = 0
        prog_ph.progress(progress_val)
        while elapsed < total_target and progress_val < 95:
            step_sleep = random.uniform(1.0, 3.0)
            if elapsed + step_sleep > total_target:
                step_sleep = total_target - elapsed
            if step_sleep > 0:
                time.sleep(step_sleep)
                elapsed += step_sleep
            inc = random.randint(5, 25)  # random increment
            progress_val = min(95, progress_val + inc)  # never decrease, cap at 95
            prog_ph.progress(progress_val)
        # --------------------------------

        try:
            # --- P2P ---
            if code == "P2P1":
                if df_vendor is None: raise RuntimeError("Vendor Master not available.")
                s.results['P2P1'] = logic6.find_missing_vendor_fields(df_vendor)

            elif code == "P2P2":
                if df_p2p is None: raise RuntimeError("P2P Sample not available.")
                if hasattr(logic6, "find_po_grn_invoice_mismatches"):
                    thr = getattr(logic6, "PO_GRN_Invoice", DEFAULTS["PO_GRN_Invoice"])
                    s.results['P2P2'] = logic6.find_po_grn_invoice_mismatches(df_p2p, variable1=thr)
                else:
                    s.results['P2P2'] = df_p2p.head(0)

            elif code == "P2P3":
                if df_p2p is None: raise RuntimeError("P2P Sample not available.")
                s.results['P2P3'] = logic6.get_invalid_rows(df_p2p) if hasattr(logic6, "get_invalid_rows") else df_p2p.head(0)

            elif code == "P2P4":
                if df_p2p is None: raise RuntimeError("P2P Sample not available.")
                s.results['P2P4'] = logic6.generate_result(df_p2p) if hasattr(logic6, "generate_result") else df_p2p.head(0)

            elif code == "P2P5":
                if df_vendor is None: raise RuntimeError("Vendor Master not available.")
                s.results['P2P5'] = logic._find_matching_rows_from_df(df_vendor) if hasattr(logic, "_find_matching_rows_from_df") else df_vendor.head(0)

            # --- O2C ---
            elif code == "O2C1":
                if df_o2c is None: raise RuntimeError("O2C Sample not available.")
                if hasattr(logic6, "check_overdue_delivery"):
                    days_thr = getattr(logic6, "OVERDUE_DAYS_THRESHOLD", DEFAULTS["OVERDUE_DAYS_THRESHOLD"])
                    dcopy = df_o2c.copy()
                    # Try signature variants: days=, then variable5=, else no arg
                    try:
                        s.results['O2C1'] = logic6.check_overdue_delivery(dcopy, days=days_thr)
                    except TypeError:
                        try:
                            s.results['O2C1'] = logic6.check_overdue_delivery(dcopy, variable5=days_thr)
                        except TypeError:
                            s.results['O2C1'] = logic6.check_overdue_delivery(dcopy)
                else:
                    s.results['O2C1'] = df_o2c.head(0)

            elif code == "O2C2":
                if df_o2c is None: raise RuntimeError("O2C Sample not available.")
                d = df_o2c.copy()
                if "Delivery_No" not in d.columns:
                    d["Delivery_No"] = pd.NA
                s.results['O2C2'] = logic6.check_dispatch_without_invoice(d) if hasattr(logic6, "check_dispatch_without_invoice") else d.head(0)

            elif code == "O2C3":
                if df_cust is None: raise RuntimeError("Customer Master not available.")
                s.results['O2C3'] = logic6.get_missing_customer_data(df_cust.copy()) if hasattr(logic6, "get_missing_customer_data") else df_cust.head(0)

            # --- H2R ---
            elif code == "H2R1":
                if (df_emp_h2r is None) or (df_att is None):
                    raise RuntimeError("H2R Employee Master or Attendance Register not available.")
                s.results['H2R1'] = logic6.find_ghost_employees(df_emp_h2r, df_att) if hasattr(logic6, "find_ghost_employees") else df_emp_h2r.head(0)

            elif code == "H2R2":
                if (df_emp_h2r is None) or (df_att is None):
                    raise RuntimeError("H2R Employee Master or Attendance Register not available.")
                if hasattr(logic6, "find_attendance_after_exit"):
                    bio2 = BytesIO()
                    with pd.ExcelWriter(bio2, engine="xlsxwriter") as writer:
                        df_emp_h2r.to_excel(writer, sheet_name="Employee_Master", index=False)
                        df_att.to_excel(writer, sheet_name="Attendance_Register", index=False)
                    bio2.seek(0)
                    s.results['H2R2'] = logic6.find_attendance_after_exit(
                        bio2, employee_sheet="Employee_Master", attendance_sheet="Attendance_Register",
                        month_col="Month", year_col=None
                    )
                else:
                    s.results['H2R2'] = df_att.head(0)

            s.proc_status[code] = "Complete"

        except Exception as e:
            s.proc_status[code] = "Failed"
            s.results[code] = pd.DataFrame({"Error": [str(e)]})

        prog_ph.progress(100)
        sym = "✅" if s.proc_status[code] == "Complete" else "❌"
        status_ph.markdown(f"- {sym} **{bot_name}** — {s.proc_status[code]}")

    # 3) Process only the selected bots
    for cat in cats:
        for code in _codes_for_category(cat):
            if sel_set and code not in sel_set:
                continue
            if s.proc_status.get(code) in ("Complete", "Failed"):
                continue
            _run_bot_and_update(code)

    # 4) Compute category statuses & finish (only for categories actually shown)
    s.statuses = {}
    for c in cats:
        codes_in_cat = [code for code in _codes_for_category(c) if (not sel_set or code in sel_set)]
        s.statuses[c] = logic6.compute_category_status(s.proc_status, codes_in_cat)

    s.processing_done = True
    _final_nav()
    st.markdown('</div>', unsafe_allow_html=True)
