# ============================== zeropage.py — Industry Selection ==============================
import streamlit as st
import streamlit.components.v1 as components
import base64
from pathlib import Path

LEFT_LOGO_PATH = "logo.png"

def _data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else ("image/jpeg" if suffix in (".jpg",".jpeg") else "image/png")
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
                  if(el && el.scrollIntoView){
                    el.scrollIntoView({block:"start", inline:"nearest"});
                  } else {
                    (window.parent||window).scrollTo(0,0);
                  }
                }catch(e){ (window.parent||window).scrollTo(0,0); }
              }
              go(); setTimeout(go, 50); setTimeout(go, 150); setTimeout(go, 300); setTimeout(go, 600);
            })();
            </script>
            """,
            height=0,
        )

st.set_page_config(
    page_title="Audit Bots • Select Industry",
    page_icon="🤖",
    layout="wide",
)

if "page" not in st.session_state:
    st.session_state.page = "zero"
if "industry" not in st.session_state:
    st.session_state.industry = None

def render_zero():
    s = st.session_state

    # Styles + header
    st.markdown(
        """
        <style>
          .ab-wrap * { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important; }
          .ab-title { font-weight:860; letter-spacing:.02em; background:linear-gradient(90deg,#1e3a8a 0%,#60a5fa 100%);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; color:transparent;
                      font-size:clamp(30px,3.2vw,38px); line-height:1.05; }
          .ab-underline { height:8px; border-radius:9999px; background:#ff7636; margin-top:8px; width:320px; max-width:30vw; }
          .ab-subtitle { margin-top:10px; font-weight:700; color:#0f172a; font-size:clamp(16px,1.6vw,20px); }
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
                <div class="ab-subtitle">Select Industry</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ab-spacer"></div>', unsafe_allow_html=True)
    _scroll_to_brand_if_needed(s)

    # Industry options
    st.subheader("Choose your industry to continue:")

    industries = [
        "-- Select --",
        "Manufacturing",
        "Banking",
        "Logistics",
        "Healthcare",
        "Retail",
        "Energy",
        "Construction",
    ]

    choice = st.selectbox("Select Industry", industries, index=0)

    # Proceed button
    if st.button("Proceed"):
        if choice == "Manufacturing":
            s.industry = "manufacturing"
            s.page = "home"   # matches firstpage router
            st.rerun()
        elif choice == "Banking":
            s.industry = "banking"
            s.page = "bank1"
            st.rerun()
        else:
            st.info("Functionality for this industry will be added soon.")

# Router
if st.session_state.page == "zero":
    render_zero()
elif st.session_state.page in ("home","process","fourth","selection","processpage","fifth","next"):
    import firstpage; firstpage.run_router()
elif st.session_state.page == "bank1":
    import b1; b1.render_bank1()
elif st.session_state.page == "bankprocess":
    import b2; b2.render_bank_process()
elif st.session_state.page == "bank3":
    import b3; b3.render_bank3()
elif st.session_state.page == "bank4":
    import b4; b4.render_bank4()
elif st.session_state.page == "bank5":
    import b5; b5.render_bank5()
elif st.session_state.page == "bank6":
    import b6; b6.render_bank6()
elif st.session_state.page == "bank7":
    import b7; b7.render_bank7()
