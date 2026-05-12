import streamlit as st
import pandas as pd
import os
import numpy as np
import sklearn.compose._column_transformer as sklearn_column_transformer
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
import joblib
from scipy import sparse
from dotenv import load_dotenv
import re
import base64
import hashlib
import io

# ─── Load environment variables ───────────────────────────────────────────────
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KrishiMitra AI — Agriculture Yield Advisor",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --soil:        #5c4a1e;
    --wheat:       #d4a843;
    --wheat-light: #f0c96a;
    --wheat-pale:  #fdf3dc;
    --leaf:        #4a7c3f;
    --leaf-bright: #6ab04c;
    --leaf-pale:   #e8f5e1;
    --text-dark:   #1c1c10;
    --text-mid:    #4a4535;
    --text-light:  #7a7260;
    --border:      #d6c99a;
    --shadow:      rgba(90,70,20,0.12);
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }

.stApp {
    background: linear-gradient(160deg, #f5f0e8 0%, #ede8db 40%, #e8f5e1 100%);
    min-height: 100vh;
}

.main .block-container {
    max-width: 720px;
    padding: 2rem 2rem 4rem;
    background: transparent;
}

.krishimitra-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}

.krishimitra-header .badge {
    display: inline-block;
    background: var(--leaf);
    color: #e8f5e1;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    margin-bottom: 0.8rem;
}

.krishimitra-header h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    color: var(--text-dark) !important;
    line-height: 1.15 !important;
    margin: 0.2rem 0 0.5rem !important;
}
.krishimitra-header h1 span { color: var(--leaf); }
.krishimitra-header p {
    font-size: 1rem;
    color: var(--text-mid);
    max-width: 460px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 300;
}

.crop-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.5rem 0 2rem;
    color: var(--wheat);
    font-size: 1.2rem;
    letter-spacing: 0.3em;
}
.crop-divider::before, .crop-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, transparent, var(--border), transparent);
}

.form-card {
    background: rgba(255,252,245,0.92);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 2rem 2rem 1.5rem;
    box-shadow: 0 4px 24px var(--shadow), 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
}

.form-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--leaf);
    margin-bottom: 1.2rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px dashed var(--border);
}

.stSelectbox label, .stNumberInput label, label[data-testid] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-mid) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.stSelectbox > div > div {
    background: #fffdf5 !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-dark) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--leaf) !important;
    box-shadow: 0 0 0 3px rgba(74,124,63,0.18) !important;
}

.stNumberInput > div > div > input {
    background: #fffdf5 !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-dark) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stNumberInput > div > div > input:focus {
    border-color: var(--leaf) !important;
    box-shadow: 0 0 0 3px rgba(74,124,63,0.18) !important;
    outline: none !important;
}
.stNumberInput button {
    background: var(--leaf-pale) !important;
    border: 1px solid var(--border) !important;
    color: var(--leaf) !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--leaf) 0%, #3d6b32 100%) !important;
    color: #f0fae8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(74,124,63,0.3) !important;
    margin-top: 0.5rem !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3d6b32 0%, #2d5224 100%) !important;
    transform: translateY(-1px) !important;
}

.yield-result {
    background: linear-gradient(135deg, #f0fae8 0%, #e2f5d4 100%);
    border: 1.5px solid var(--leaf-bright);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    text-align: center;
    box-shadow: 0 4px 20px rgba(74,124,63,0.15);
}
.yield-result .yield-label {
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--leaf);
}
.yield-result .yield-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--leaf);
    line-height: 1.1;
}
.yield-result .yield-unit {
    font-size: 1rem;
    color: var(--text-mid);
    font-weight: 300;
}

.advisor-card {
    background: rgba(255,252,245,0.95);
    border: 1px solid var(--border);
    border-left: 4px solid var(--wheat);
    border-radius: 0 14px 14px 0;
    padding: 1.5rem 1.75rem;
    margin-top: 1.2rem;
    box-shadow: 0 2px 12px var(--shadow);
}
.advisor-card .advisor-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    color: var(--soil);
    margin-bottom: 0.8rem;
    font-weight: 500;
}

.stAlert { border-radius: 12px !important; }

.krishimitra-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: var(--text-light);
    font-size: 0.78rem;
}
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def clean_for_tts(text: str) -> str:
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s?', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'\n+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def translate_to_hindi(text: str) -> str:
    """Use Google Translate free REST endpoint — no extra packages needed."""
    try:
        import urllib.request, urllib.parse, json
        url = (
            "https://translate.googleapis.com/translate_a/single"
            "?client=gtx&sl=en&tl=hi&dt=t&q="
            + urllib.parse.quote(text[:4500])   # API cap
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        return "".join(seg[0] for seg in data[0] if seg[0])
    except Exception:
        return text                             # fallback: speak English in Hindi voice


def make_audio_b64(text: str, lang: str) -> str | None:
    """Generate MP3 with gTTS and return base64 string."""
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return None


# ─── TTS Player ───────────────────────────────────────────────────────────────
def render_tts_player(advisor_text: str):
    clean = clean_for_tts(advisor_text)
    cache_key = hashlib.md5(clean.encode()).hexdigest()[:10]
    en_key = f"tts_en_{cache_key}"
    hi_key = f"tts_hi_{cache_key}"

    # Generate English audio once
    if en_key not in st.session_state:
        with st.spinner("🔊 Preparing audio..."):
            st.session_state[en_key] = make_audio_b64(clean, "en") or ""

    en_b64  = st.session_state.get(en_key, "")
    hi_b64  = st.session_state.get(hi_key, "")

    # ── Player HTML ──────────────────────────────────────────────────────────
    player_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: 'DM Sans', sans-serif; }}

  .player {{
    background: linear-gradient(145deg, #fffdf5, #f5f0e0);
    border: 1.5px solid #d6c99a;
    border-radius: 18px;
    padding: 1.1rem 1.4rem 1rem;
    box-shadow: 0 3px 16px rgba(90,70,20,0.1), inset 0 1px 0 rgba(255,255,255,0.8);
  }}

  /* ── Top row ── */
  .top-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
  }}
  .player-label {{
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }}
  .label-icon {{
    width: 30px; height: 30px;
    background: linear-gradient(135deg, #4a7c3f, #3d6b32);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem;
  }}
  .label-text {{
    font-size: 0.75rem;
    font-weight: 500;
    color: #5c4a1e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}
  .lang-pills {{
    display: flex;
    background: #ede8d8;
    border-radius: 20px;
    padding: 3px;
    gap: 2px;
  }}
  .lang-pill {{
    padding: 0.22rem 0.75rem;
    border-radius: 16px;
    font-size: 0.75rem;
    font-weight: 500;
    cursor: pointer;
    border: none;
    background: transparent;
    color: #7a7260;
    transition: all 0.2s;
    font-family: 'DM Sans', sans-serif;
  }}
  .lang-pill.active {{
    background: white;
    color: #4a7c3f;
    box-shadow: 0 1px 4px rgba(90,70,20,0.15);
  }}

  /* ── Controls row ── */
  .controls-row {{
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.85rem;
  }}
  .play-btn {{
    width: 44px; height: 44px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4a7c3f, #3d6b32);
    border: none; color: white;
    font-size: 1rem; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 3px 12px rgba(74,124,63,0.4);
    transition: all 0.2s; flex-shrink: 0;
  }}
  .play-btn:hover {{ transform: scale(1.08); }}
  .play-btn:active {{ transform: scale(0.95); }}
  .stop-btn {{
    width: 34px; height: 34px;
    border-radius: 50%;
    background: #fff0f0;
    border: 1.5px solid #e9b4b4;
    color: #b33; font-size: 0.75rem;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s; flex-shrink: 0;
  }}
  .stop-btn:hover {{ background: #ffe0e0; }}

  /* ── Waveform ── */
  .wave-wrap {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }}
  .waveform {{
    display: flex;
    align-items: center;
    gap: 2.5px;
    height: 30px;
  }}
  .bar {{
    width: 3px;
    border-radius: 2px;
    background: #d6c99a;
    transition: background 0.2s;
  }}
  .bar.playing {{
    background: linear-gradient(to top, #4a7c3f, #6ab04c);
  }}
  @keyframes wa {{ 0%,100%{{height:5px}} 50%{{height:22px}} }}
  @keyframes wb {{ 0%,100%{{height:9px}} 50%{{height:14px}} }}
  @keyframes wc {{ 0%,100%{{height:14px}} 50%{{height:7px}} }}
  @keyframes wd {{ 0%,100%{{height:7px}} 50%{{height:20px}} }}
  @keyframes we {{ 0%,100%{{height:18px}} 50%{{height:5px}} }}

  /* ── Progress ── */
  .progress-track {{
    height: 4px;
    background: #e8e0cc;
    border-radius: 2px;
    overflow: hidden;
  }}
  .progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, #4a7c3f, #d4a843);
    border-radius: 2px;
    width: 0%;
    transition: width 0.4s linear;
  }}

  /* ── Bottom row ── */
  .bottom-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 0.6rem;
  }}
  .status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: #f0fae8;
    border: 1px solid #b8dfa8;
    border-radius: 20px;
    padding: 0.18rem 0.65rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: #4a7c3f;
    min-width: 78px;
    justify-content: center;
    transition: all 0.3s;
  }}
  .status-pill.error   {{ background:#fff0f0; border-color:#e9b4b4; color:#b33; }}
  .status-pill.loading {{ background:#fdf3dc; border-color:#e8c97a; color:#5c4a1e; }}
  .status-pill.done    {{ background:#e8f5e1; border-color:#6ab04c; color:#3d6b32; }}

  .speed-row {{
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }}
  .speed-label {{ font-size: 0.7rem; color: #7a7260; }}
  .speed-slider {{
    width: 70px; height: 4px;
    accent-color: #4a7c3f;
    cursor: pointer;
  }}
  .speed-val {{ font-size: 0.7rem; font-weight: 500; color: #4a4535; min-width: 25px; }}

  /* ── Hindi loading overlay ── */
  .hi-loading {{
    display: none;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.6rem;
    background: #fdf3dc;
    border: 1px dashed #d4a843;
    border-radius: 10px;
    padding: 0.5rem 0.8rem;
    font-size: 0.76rem;
    color: #5c4a1e;
  }}
  .spinner {{
    width: 14px; height: 14px;
    border: 2px solid #d4a843;
    border-top-color: #4a7c3f;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
<div class="player">

  <!-- Top row -->
  <div class="top-row">
    <div class="player-label">
      <div class="label-icon">🔊</div>
      <span class="label-text">Listen to Advisor</span>
    </div>
    <div class="lang-pills">
      <button class="lang-pill active" id="pillEN" onclick="switchLang('en')">🇬🇧 EN</button>
      <button class="lang-pill"        id="pillHI" onclick="switchLang('hi')">🇮🇳 हि</button>
    </div>
  </div>

  <!-- Controls -->
  <div class="controls-row">
    <button class="play-btn" id="playBtn" onclick="togglePlay()">▶</button>
    <button class="stop-btn" id="stopBtn" onclick="stopAll()">■</button>
    <div class="wave-wrap">
      <div class="waveform" id="waveform"></div>
      <div class="progress-track"><div class="progress-fill" id="pFill"></div></div>
    </div>
  </div>

  <!-- Bottom row -->
  <div class="bottom-row">
    <div class="status-pill" id="statusPill">⏸ Ready</div>
    <div class="speed-row">
      <span class="speed-label">Speed</span>
      <input type="range" class="speed-slider" id="speedSlider"
             min="0.5" max="2" step="0.1" value="1"
             oninput="onSpeed(this.value)">
      <span class="speed-val" id="speedVal">1x</span>
    </div>
  </div>

  <!-- Hindi loading notice -->
  <div class="hi-loading" id="hiLoading">
    <div class="spinner"></div>
    <span id="hiMsg">Generating Hindi audio…</span>
  </div>

</div>

<script>
// ── Waveform bars ─────────────────────────────────────────────────────────────
var anims  = ['wa','wb','wc','wd','we'];
var bHeights = [5,10,16,8,20,6,14,22,9,13,18,7,17,11,5,21,15,9,23,7,13,17,5,19,11];
var wf = document.getElementById('waveform');
bHeights.forEach(function(h, i) {{
  var d = document.createElement('div');
  d.className = 'bar';
  d.id = 'bar' + i;
  d.style.height = h + 'px';
  d.dataset.base = h;
  d.dataset.anim = anims[i % 5];
  wf.appendChild(d);
}});

// ── State ─────────────────────────────────────────────────────────────────────
var enAudio  = null;
var hiAudio  = null;
var cur      = null;
var lang     = 'en';
var playing  = false;
var pTimer   = null;

var enB64 = "{en_b64}";
var hiB64 = "{hi_b64}";

function mkAudio(b64) {{
  if (!b64) return null;
  var a = new Audio("data:audio/mp3;base64," + b64);
  a.onended  = onEnd;
  a.onerror  = onErr;
  return a;
}}

if (enB64) enAudio = mkAudio(enB64);
if (hiB64) hiAudio = mkAudio(hiB64);
cur = enAudio;

// ── UI helpers ────────────────────────────────────────────────────────────────
function setStatus(txt, cls) {{
  var el = document.getElementById('statusPill');
  el.textContent = txt;
  el.className = 'status-pill' + (cls ? ' ' + cls : '');
}}
function setProgress(p) {{
  document.getElementById('pFill').style.width = Math.min(p,100) + '%';
}}
function startWave() {{
  bHeights.forEach(function(_, i) {{
    var b = document.getElementById('bar' + i);
    b.classList.add('playing');
    var dur = (0.38 + (i%5)*0.11).toFixed(2);
    b.style.animation = b.dataset.anim + ' ' + dur + 's ease-in-out infinite';
    b.style.animationDelay = (i*0.038).toFixed(3) + 's';
  }});
}}
function stopWave() {{
  bHeights.forEach(function(h, i) {{
    var b = document.getElementById('bar' + i);
    b.classList.remove('playing');
    b.style.animation = 'none';
    b.style.height = h + 'px';
  }});
}}
function startTimer(dur) {{
  clearInterval(pTimer);
  var t0 = Date.now();
  pTimer = setInterval(function() {{
    var elapsed = (Date.now()-t0)/1000;
    var spd = parseFloat(document.getElementById('speedSlider').value);
    var adjDur = dur / spd;
    var pct = adjDur > 0 ? (elapsed/adjDur)*100 : 0;
    setProgress(pct);
    var rem = Math.max(0, Math.round(adjDur - elapsed));
    setStatus('🔊 ' + rem + 's left');
    if (pct >= 100) clearInterval(pTimer);
  }}, 400);
}}

// ── Events ────────────────────────────────────────────────────────────────────
function onEnd() {{
  playing = false;
  clearInterval(pTimer);
  stopWave();
  setProgress(100);
  setStatus('✓ Done', 'done');
  document.getElementById('playBtn').textContent = '▶';
  setTimeout(function() {{ setProgress(0); setStatus('⏸ Ready'); }}, 2500);
}}
function onErr() {{
  playing = false;
  stopWave();
  clearInterval(pTimer);
  setStatus('✗ Error', 'error');
  document.getElementById('playBtn').textContent = '▶';
}}

// ── Controls ──────────────────────────────────────────────────────────────────
function togglePlay() {{
  if (!cur) {{ setStatus('No audio', 'error'); return; }}
  var spd = parseFloat(document.getElementById('speedSlider').value);
  cur.playbackRate = spd;
  if (playing) {{
    cur.pause();
    playing = false;
    clearInterval(pTimer);
    stopWave();
    setStatus('⏸ Paused');
    document.getElementById('playBtn').textContent = '▶';
  }} else {{
    cur.play().then(function() {{
      playing = true;
      startWave();
      document.getElementById('playBtn').textContent = '⏸';
      var dur = cur.duration && isFinite(cur.duration) ? cur.duration : 90;
      startTimer(dur);
      setStatus('🔊 Playing');
    }}).catch(onErr);
  }}
}}

function stopAll() {{
  if (cur) {{ cur.pause(); cur.currentTime = 0; }}
  playing = false;
  clearInterval(pTimer);
  stopWave();
  setProgress(0);
  setStatus('⏸ Ready');
  document.getElementById('playBtn').textContent = '▶';
}}

function onSpeed(v) {{
  document.getElementById('speedVal').textContent = parseFloat(v).toFixed(1) + 'x';
  if (cur) cur.playbackRate = parseFloat(v);
}}

function switchLang(l) {{
  stopAll();
  lang = l;
  document.getElementById('pillEN').classList.toggle('active', l==='en');
  document.getElementById('pillHI').classList.toggle('active', l==='hi');

  if (l === 'en') {{
    cur = enAudio;
    document.getElementById('hiLoading').style.display = 'none';
    setStatus('⏸ Ready');
  }} else {{
    if (hiAudio) {{
      cur = hiAudio;
      setStatus('⏸ Ready');
    }} else {{
      cur = null;
      document.getElementById('hiLoading').style.display = 'flex';
      document.getElementById('hiMsg').textContent = 'Hindi audio not generated yet. Click the button below.';
      setStatus('⏳ Pending', 'loading');
    }}
  }}
}}
</script>
</body>
</html>
"""
    st.components.v1.html(player_html, height=195)

    # ── Hindi generation button (outside the iframe) ──────────────────────────
    if not hi_b64:
        if st.button("🇮🇳 Generate Hindi Audio", key=f"gen_hi_{cache_key}", use_container_width=True):
            with st.spinner("🔄 Translating to Hindi and generating audio…"):
                hindi_text = translate_to_hindi(clean)
                audio      = make_audio_b64(hindi_text, "hi")
            if audio:
                st.session_state[hi_key] = audio
                st.success("✅ Hindi audio ready! Switch to 🇮🇳 हि and press ▶ Play")
                st.rerun()
            else:
                st.error("❌ gTTS failed. Make sure you have internet access and `gtts` installed.")


# ─── Sklearn Patch ────────────────────────────────────────────────────────────
def apply_sklearn_pickle_compatibility_patch():
    if not hasattr(sklearn_column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        sklearn_column_transformer._RemainderColsList = _RemainderColsList


# ─── Initialize LLM ───────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key) if groq_api_key else None

# ─── Session States ───────────────────────────────────────────────────────────
for key in ["history", "predicted_yield", "llm_explanation", "yield_pred"]:
    if key not in st.session_state:
        st.session_state[key] = ChatMessageHistory() if key == "history" else None


# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline_model():
    apply_sklearn_pickle_compatibility_patch()
    return joblib.load("crop_yield.joblib")


@st.cache_data
def load_dataset():
    return pd.read_csv("new_df.csv")


try:
    pipeline = load_pipeline_model()
except Exception as exc:
    st.error(f"⚠️ Model could not be loaded: {exc}")
    st.info("Try: pip install scikit-learn==1.6.1")
    st.stop()

df          = load_dataset()
crop_list   = sorted(df["Crop"].dropna().unique())
state_list  = sorted(df["State"].dropna().unique())
season_list = sorted(df["Season"].dropna().unique())

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="krishimitra-header">
    <div class="badge">🌱 AI-Powered Agriculture</div>
    <h1>Krishi<span>Mitra</span> AI</h1>
    <p>Predict crop yield with precision. Get expert advice tailored to your farm's conditions.</p>
</div>
<div class="crop-divider">🌾 🌾 🌾</div>
""", unsafe_allow_html=True)

# ─── Form: Crop & Location ────────────────────────────────────────────────────
st.markdown('<div class="form-card"><div class="form-card-title">🌿 Crop & Location</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    crop = st.selectbox("Crop", crop_list)
with col2:
    season = st.selectbox("Season", season_list)
state = st.selectbox("State / Region", state_list)
st.markdown("</div>", unsafe_allow_html=True)

# ─── Form: Farm Inputs ────────────────────────────────────────────────────────
st.markdown('<div class="form-card"><div class="form-card-title">🚜 Farm Inputs & Conditions</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    area       = st.number_input("Cultivated Area (hectares)", min_value=0.0, step=0.5,  value=5.0)
    fertilizer = st.number_input("Fertilizer Used (kg)",       min_value=0.0, step=1.0,  value=5.0)
with col4:
    rainfall   = st.number_input("Annual Rainfall (mm)",       min_value=0.0, step=10.0, value=100.0)
    pesticide  = st.number_input("Pesticide Used (kg)",        min_value=0.0, step=0.5,  value=2.0)
st.markdown("</div>", unsafe_allow_html=True)


# ─── Predict ──────────────────────────────────────────────────────────────────
def predict_yield():
    input_data = pd.DataFrame({
        "Season":          [season],
        "State":           [state],
        "Annual_Rainfall": [rainfall],
        "Fertilizer":      [fertilizer],
        "Pesticide":       [pesticide],
        "Crop":            [crop],
        "Area":            [area],
    })
    try:
        return pipeline.predict(input_data)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


if st.button("🌾 Predict Yield"):
    with st.spinner("Analysing your farm data..."):
        pred = predict_yield()

    if pred is not None:
        st.markdown(f"""
        <div class="yield-result">
            <div class="yield-label">Estimated Yield</div>
            <div class="yield-value">{pred:,.2f}</div>
            <div class="yield-unit">tonnes per hectare</div>
        </div>
        """, unsafe_allow_html=True)

        if llm:
            with st.spinner("🌿 Consulting your AI farm advisor..."):
                prompt = f"""
You are KrishiMitra, a friendly and knowledgeable agriculture expert for Indian farmers.

Farmer's details:
- Crop: {crop} | Season: {season} | State: {state}
- Area: {area} ha | Rainfall: {rainfall} mm
- Fertilizer: {fertilizer} kg | Pesticide: {pesticide} kg
- Predicted Yield: {pred:.2f} tonnes

Provide a warm, practical response:
1. 🌱 **Why this yield?** — Key factors affecting this result.
2. 🚜 **How to improve?** — 3 actionable tips for next season.
3. 🏛️ **Govt. Schemes** — 2-3 relevant Indian government schemes.

Keep it simple and farmer-friendly. Use bullet points.
"""
                try:
                    response = llm.invoke(prompt)
                    st.session_state.llm_explanation = response.content
                    # Clear old TTS cache on new prediction
                    for k in list(st.session_state.keys()):
                        if k.startswith("tts_"):
                            del st.session_state[k]
                except Exception as e:
                    st.error(f"Advisor error: {e}")
        else:
            st.warning("💡 Add your GROQ_API_KEY to .env to unlock AI-powered farm advice.")

# ─── Advisor + TTS ────────────────────────────────────────────────────────────
if st.session_state.llm_explanation:
    st.markdown("""
    <div class="advisor-card">
        <div class="advisor-header">🤖 Your KrishiMitra Advisor Says:</div>
    </div>
    """, unsafe_allow_html=True)
    #st.markdown(st.session_state.llm_explanation)
    st.markdown(
    f"""
    <div style="
        color:#2E4A1F;
        font-size:18px;
        line-height:1.8;
        font-weight:500;
    ">
    {st.session_state.llm_explanation}
    </div>
    """,
    unsafe_allow_html=True
)
    render_tts_player(st.session_state.llm_explanation)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="krishimitra-footer">
    🌾 KrishiMitra AI · Empowering Indian Farmers with Technology<br>
    <span style="font-size:0.72rem; opacity:0.7;">Built with Streamlit & LangChain · Powered by Groq LLaMA</span>
</div>
""", unsafe_allow_html=True)