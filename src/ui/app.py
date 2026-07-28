"""
app.py — DrishtAI Stage 8 (Streamlit review console)

Upload footage, run the full pipeline, and get the answer to the question a
reviewer actually asks: WHEN did it happen, and what led to it.

Run:
    streamlit run src/ui/app.py

Design decisions:

1. LIVE UPLOAD IS THE PRIMARY PATH. A judge handing over unseen footage and
   getting an answer is the demo. Pre-computed results would be faster but
   would not prove anything.

2. HONEST PROGRESS, NOT A SPINNER. Detection on CPU is genuinely slow — a
   29 s clip is ~435 processed frames through YOLOv8s. A spinner with no
   numbers makes 3 minutes feel broken. We show the stage, the frame count,
   and a live ETA computed from measured throughput, so the wait is
   legible.

3. ANALYSIS QUALITY IS A USER CHOICE, STATED IN PLAIN TERMS. The presets
   trade accuracy for time. They are labelled by what they do for the
   viewer ("Quick look", "Balanced", "Thorough"), not by model names, and
   each states its real cost.

4. EVERY STAGE IS THE PIPELINE MODULE, NOT A REIMPLEMENTATION. The UI
   imports tracker/motion_math/collision_detector/timeline_builder/explain
   and calls them. Nothing about the analysis lives in this file, so the UI
   can never drift from what the command line produces.

5. THE UI DEGRADES, IT DOES NOT CRASH. No collision found, no vehicles
   found, no API key, unreadable video — each has a written state that says
   what happened and what to do next.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

# --- make the pipeline modules importable regardless of where streamlit runs
ROOT = Path(__file__).resolve().parents[2]
for sub in ("detection", "motion", "reasoning", "timeline"):
    p = ROOT / "src" / sub
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tracker import VehicleTracker                      # noqa: E402
from motion_math import run as motion_run               # noqa: E402
from collision_detector import run as collision_run     # noqa: E402
from timeline_builder import build_timeline             # noqa: E402
from explain import (                                   # noqa: E402
    explain_with_api, explain_offline, answer_query, explain_cause,
    humanise_seconds, seconds_of,
)
from vehicle_labels import (                            # noqa: E402
    build_class_map, humanise_ids, label_for,
)

st.set_page_config(page_title="DrishtAI — Incident Review",
                   page_icon="◉", layout="wide")

# --- Palette: ink on paper, two signal colours that MEAN something.
#     amber = the earliest observable warning, red = the impact.
#     Nothing else is coloured, so colour always carries information.
INK, MUTED, RULE = "#1B1F24", "#6B7280", "#E3E6EA"
WARN, IMPACT, CALM = "#B45309", "#B91C1C", "#3F6212"

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>

/* -------------------------------------------------------
   GLOBAL
--------------------------------------------------------*/

:root{

--bg:#070B14;
--sidebar:#0B1120;
--card:#111827;
--card2:#161F2F;

--border:#243244;

--text:#F8FAFC;
--muted:#94A3B8;

--blue:#3B82F6;
--cyan:#22D3EE;

--green:#22C55E;
--orange:#F59E0B;
--red:#EF4444;

}

html,
body,
[class*="css"]{

font-family:"Inter",sans-serif;
background:var(--bg);
color:var(--text);

}
.stTextInput input{

background:#111827 !important;

border:2px solid rgba(59,130,246,.15) !important;

border-radius:18px !important;

height:62px !important;

font-size:18px !important;

padding-left:18px !important;

transition:.25s;

}

.stTextInput input:focus{

border:2px solid #22D3EE !important;

box-shadow:0 0 25px rgba(34,211,238,.25);

}

/* page */

.stApp{

background:
radial-gradient(circle at top right,#13203d 0%,transparent 28%),
radial-gradient(circle at bottom left,#101b35 0%,transparent 25%),
var(--bg);

}

/* -------------------------------------------------------
 SIDEBAR
--------------------------------------------------------*/

[data-testid="stSidebar"]{

background:var(--sidebar);
border-right:1px solid var(--border);

}

[data-testid="stSidebar"] *{

color:white;

}

[data-testid="stSidebar"] label{

font-weight:600;

}

[data-testid="stSidebar"] .stRadio{

padding-top:10px;

}

/* -------------------------------------------------------
 HEADINGS
--------------------------------------------------------*/

.eyebrow{

color:var(--cyan);

font-size:.75rem;

letter-spacing:.18em;

text-transform:uppercase;

font-weight:700;

margin-bottom:12px;

}

.headline{

font-family:"Space Grotesk",sans-serif;

font-size:4rem;

font-weight:700;

line-height:1;

color:white;

margin-bottom:8px;

}

.sub{

color:var(--muted);

font-size:1.1rem;

line-height:1.7;

}

/* -------------------------------------------------------
 CARDS
--------------------------------------------------------*/

.card{

background:rgba(17,24,39,.88);

backdrop-filter:blur(18px);

border:1px solid rgba(255,255,255,.08);

border-radius:22px;

padding:28px;

box-shadow:

0 0 0 1px rgba(255,255,255,.03),

0 20px 60px rgba(0,0,0,.35);

transition:.25s;

}

.card:hover{

transform:translateY(-3px);

box-shadow:

0 0 35px rgba(59,130,246,.15),

0 20px 60px rgba(0,0,0,.45);

}

/* -------------------------------------------------------
 BIG NUMBERS
--------------------------------------------------------*/

.answer{

font-family:"Space Grotesk";

font-size:4.5rem;

font-weight:700;

color:#ff5d5d;

}

.answer-sub{

font-family:monospace;

font-size:1rem;

color:#CBD5E1;

margin-top:10px;

}

.lead{

font-family:"Space Grotesk";

font-size:4rem;

font-weight:700;

color:#F59E0B;

}

/* -------------------------------------------------------
 BUTTONS
--------------------------------------------------------*/

.stButton>button{

background:

linear-gradient(
135deg,
#2563EB,
#06B6D4
);

border:none;

border-radius:14px;

height:55px;

font-size:16px;

font-weight:600;

color:white;

transition:.25s;

}

.stButton>button:hover{

transform:translateY(-2px);

box-shadow:

0 0 20px rgba(34,211,238,.35);

}

/* -------------------------------------------------------
 INPUTS
--------------------------------------------------------*/

.stTextInput input{

background:#0E1726;

border:1px solid var(--border);

color:white;

border-radius:14px;

}

.stSelectbox div{

background:#0E1726;

}

/* -------------------------------------------------------
 FILE UPLOADER
--------------------------------------------------------*/

[data-testid="stFileUploader"]{

background:#101826;

border:2px dashed #2B3B50;

border-radius:22px;

padding:35px;

}

[data-testid="stFileUploader"]:hover{

border-color:#22D3EE;

}

/* -------------------------------------------------------
 METRICS
--------------------------------------------------------*/

[data-testid="metric-container"]{

background:#111827;

border:1px solid rgba(255,255,255,.08);

border-radius:18px;

padding:18px;

}

[data-testid="stMetricLabel"]{

color:#94A3B8;

}

[data-testid="stMetricValue"]{

color:white;

font-size:2rem;

}

/* -------------------------------------------------------
 EXPANDER
--------------------------------------------------------*/

.streamlit-expanderHeader{

font-weight:600;

color:white;

}

/* -------------------------------------------------------
 TABLES
--------------------------------------------------------*/

thead tr{

background:#0F172A;

}

tbody tr{

background:#111827;

}

/* -------------------------------------------------------
 SCROLLBAR
--------------------------------------------------------*/

::-webkit-scrollbar{

width:10px;

}

::-webkit-scrollbar-track{

background:#0B1120;

}

::-webkit-scrollbar-thumb{

background:#334155;

border-radius:20px;

}

::-webkit-scrollbar-thumb:hover{

background:#475569;

}

/* -------------------------------------------------------
 DIVIDER
--------------------------------------------------------*/

hr{

border:none;

height:1px;

background:rgba(255,255,255,.08);

margin:40px 0;

}

</style>

""", unsafe_allow_html=True)

st.markdown("""
<div style="
position:relative;
top:0;
z-index:999;
background:rgba(7,11,20,.85);
backdrop-filter:blur(18px);
border-bottom:1px solid rgba(255,255,255,.06);
padding:18px 30px;
margin-bottom:30px;
">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
">

<div style="
font-family:'Space Grotesk';
font-size:26px;
font-weight:700;
color:white;
">
🚨 DrishtAI
</div>

<div style="
display:flex;
gap:14px;
">
<span style="color:#94A3B8;">Dashboard</span>
<span style="color:#94A3B8;">Timeline</span>
<span style="color:#94A3B8;">Evidence</span>
<span style="color:#94A3B8;">AI Report</span>
</div>

</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Presets — stated as what they cost the viewer, not as model names
# ---------------------------------------------------------------------------
PRESETS = {
    "Quick look": dict(model="yolov8n.pt", interval=3, imgsz=480,
                       note="Fastest. May miss vehicles during the impact."),
    "Balanced": dict(model="yolov8s.pt", interval=2, imgsz=640,
                     note="Recommended. What our results were validated on."),
    "Thorough": dict(model="yolov8s.pt", interval=1, imgsz=640,
                     note="Every frame. Best timing precision, ~2x the wait."),
}


def probe(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("This file could not be opened as video. "
                           "Try re-saving it as MP4 (H.264).")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0:
        raise RuntimeError("This video reports no frame rate. Re-encode it "
                           "with: ffmpeg -i in.mp4 -r 30 out.mp4")
    return dict(fps=fps, frames=n, width=w, height=h, duration=n / fps)


def grab_frame(path: str, frame_index: int):
    """Read one specific frame for the evidence still."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def draw_boxes(frame, records: list[dict], ids: list[str]):
    """Mark the two vehicles that collided, and only those two."""
    import numpy as np
    out = np.ascontiguousarray(frame.copy())
    for r in records:
        if r["object_id"] not in ids:
            continue
        x1, y1, x2, y2 = map(int, r["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (185, 28, 28), 3)
        cv2.putText(out, r["object_id"], (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (185, 28, 28), 2)
    return out


def time_ruler(duration: float, warn_s: float | None, impact_s: float | None,
               impact_label: str = "impact"):
    """
    SIGNATURE ELEMENT — the clip drawn to scale, with the two moments that
    matter marked and the lead-time gap drawn between them.

    The project's whole claim is "we saw it coming N seconds early". A
    number in a table states that; a gap drawn to scale lets a viewer see
    how much time a driver or operator actually had.
    """
    W, H = 1000, 108
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="Clip timeline with warning and impact marked">',
             f'<line x1="0" y1="62" x2="{W}" y2="62" stroke="{RULE}" stroke-width="2"/>']

    step = 1 if duration <= 10 else (5 if duration <= 60 else 15)
    t = 0.0
    while t <= duration:
        x = (t / duration) * W if duration else 0
        parts.append(f'<line x1="{x:.1f}" y1="56" x2="{x:.1f}" y2="68" '
                     f'stroke="{RULE}" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="86" font-size="11" fill="{MUTED}" '
                     f'text-anchor="middle" font-family="monospace">{t:g}s</text>')
        t += step

    if warn_s is not None and impact_s is not None and duration:
        xw, xi = (warn_s / duration) * W, (impact_s / duration) * W
        parts.append(f'<rect x="{xw:.1f}" y="54" width="{max(xi-xw,2):.1f}" '
                     f'height="16" fill="{WARN}" opacity="0.16"/>')
        parts.append(f'<text x="{(xw+xi)/2:.1f}" y="30" font-size="13" '
                     f'fill="{WARN}" text-anchor="middle" font-weight="700">'
                     f'{impact_s - warn_s:.2f}s of warning</text>')
        parts.append(f'<line x1="{xw:.1f}" y1="34" x2="{xi:.1f}" y2="34" '
                     f'stroke="{WARN}" stroke-width="1.5"/>')

    for val, colour, label in ((warn_s, WARN, "warning"),
                               (impact_s, IMPACT, impact_label)):
        if val is None or not duration:
            continue
        x = (val / duration) * W
        parts.append(f'<line x1="{x:.1f}" y1="42" x2="{x:.1f}" y2="82" '
                     f'stroke="{colour}" stroke-width="3"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="42" r="5" fill="{colour}"/>')
        parts.append(f'<text x="{x:.1f}" y="104" font-size="11" fill="{colour}" '
                     f'text-anchor="middle" font-weight="600">{label}</text>')

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def analyse(path: str, cfg: dict, ui) -> dict:
    """Run stages 3-6. `ui` receives (stage_no, label, fraction, detail)."""
    meta = probe(path)
    to_process = meta["frames"] // cfg["interval"]

    ui(1, "Detecting and tracking vehicles", 0.0,
       f"{to_process} frames to process")

    tracker = VehicleTracker(cfg["model"], 0.10)
    cap = cv2.VideoCapture(path)
    fps = meta["fps"]
    records, idx, done, t0 = [], 0, 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % cfg["interval"] == 0:
            res = tracker.model.track(frame, conf=0.10,
                                      classes=[2, 3, 5, 7],
                                      tracker="bytetrack.yaml",
                                      persist=True, imgsz=cfg["imgsz"],
                                      verbose=False)
            boxes = res[0].boxes
            if boxes.id is not None:
                from tracker import frame_index_to_timestamp
                ts = frame_index_to_timestamp(idx, fps)
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    records.append(dict(
                        object_id=f"vehicle_{int(boxes.id[i].item())}",
                        frame_index=idx, timestamp=ts, time_seconds=idx / fps,
                        bbox=[round(v, 1) for v in (x1, y1, x2, y2)],
                        position=[round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                        confidence=round(float(boxes.conf[i].item()), 3),
                        vehicle_class={2: "car", 3: "motorcycle",
                                       5: "bus", 7: "truck"}[int(boxes.cls[i].item())],
                    ))
            done += 1
            if done % 5 == 0 or done == to_process:
                rate = done / max(time.time() - t0, .001)
                left = max(to_process - done, 0) / max(rate, .001)
                ui(1, "Detecting and tracking vehicles", done / max(to_process, 1),
                   f"frame {done} of {to_process} · about {left:.0f}s remaining")
        idx += 1
    cap.release()

    if not records:
        return dict(meta=meta, tracks=[], motion=[], ranked=[],
                    events=[], timeline=[], class_map={},
                    reason="no_vehicles")

    # One vote per tracked vehicle, over every frame it appeared in. Done
    # here rather than at render time so the whole app reads one decision
    # instead of each panel re-deriving its own from whatever records it
    # happens to hold.
    class_map = build_class_map(records)

    ui(2, "Measuring speed and direction", 1.0,
       f"{len({r['object_id'] for r in records})} vehicles tracked")
    motion_recs, prov, _ = motion_run(records, smooth_window=3, do_merge=True)
    motion = [r.__dict__ for r in motion_recs]

    ui(3, "Looking for collisions", 1.0, "comparing every vehicle pair")
    ranked, events = collision_run(motion, window=3, top=5)

    if not events:
        return dict(meta=meta, tracks=records, motion=motion, ranked=ranked,
                    events=[], timeline=[], merged=prov,
                    class_map=class_map, reason="no_collision")

    ui(4, "Building the event timeline", 1.0, "finding the earliest warning")
    tl_recs, anchor, warning, lead = build_timeline(
        [e.__dict__ if hasattr(e, "__dict__") else e for e in events])

    return dict(meta=meta, tracks=records, motion=motion, ranked=ranked,
                events=[e.__dict__ for e in events],
                timeline=[r.__dict__ for r in tl_recs],
                anchor=anchor, warning=warning, lead=lead,
                merged=prov, class_map=class_map, reason="ok")


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.markdown("""

<div style="padding-top:25px;padding-bottom:35px;">

<div style="display:flex;
align-items:center;
justify-content:space-between;">

<div>

<div style="
font-family:'Space Grotesk';
font-size:16px;
letter-spacing:4px;
color:#22D3EE;
font-weight:700;
">

DRISHTAI AI PLATFORM

</div>

<div style="
font-family:'Space Grotesk';
font-size:62px;
font-weight:700;
line-height:1.05;
color:white;
margin-top:8px;
">

AI Powered<br>
Accident Investigation

</div>

<div style="
margin-top:22px;
font-size:18px;
line-height:1.8;
max-width:700px;
color:#94A3B8;
">

Upload CCTV or dashcam footage and let DrishtAI automatically detect
collisions, reconstruct vehicle movement, estimate the earliest warning,
and generate an AI-powered investigation report.

</div>

<div style="margin-top:35px;display:flex;gap:18px;flex-wrap:wrap;">

<div style="
padding:10px 18px;
border-radius:999px;
background:#132238;
border:1px solid #1E3A5F;
color:#38BDF8;
font-weight:600;
">

🚗 Vehicle Tracking

</div>

<div style="
padding:10px 18px;
border-radius:999px;
background:#132238;
border:1px solid #1E3A5F;
color:#38BDF8;
font-weight:600;
">

⚡ Motion Analysis

</div>

<div style="
padding:10px 18px;
border-radius:999px;
background:#132238;
border:1px solid #1E3A5F;
color:#38BDF8;
font-weight:600;
">

🚨 Collision Detection

</div>

<div style="
padding:10px 18px;
border-radius:999px;
background:#132238;
border:1px solid #1E3A5F;
color:#38BDF8;
font-weight:600;
">

🤖 AI Report

</div>

</div>

</div>

<div>

<div style="
width:280px;
height:280px;
border-radius:28px;
background:linear-gradient(135deg, rgba(59,130,246,.15), rgba(34,211,238,.08));
display:flex;
justify-content:center;
align-items:center;
border:1px solid rgba(255,255,255,.08);
box-shadow:0 0 60px rgba(59,130,246,.18);
">

<div style="text-align:center;">

<div style="
font-size:90px;
">

🛰️

</div>

<div style="
font-family:'Space Grotesk';
font-size:30px;
font-weight:700;
color:white;
">

DrishtAI

</div>

<div style="
color:#94A3B8;
margin-top:6px;
">

Incident Intelligence

</div>

</div>

</div>

</div>

</div>

</div>

""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="margin-top:10px;margin-bottom:30px;">

    <div style="
    font-family:'Space Grotesk';
    font-size:30px;
    font-weight:700;
    color:white;
    ">

    🚨 DrishtAI

    </div>

    <div style="
    color:#94A3B8;
    margin-top:8px;
    ">

    AI Investigation Dashboard

    </div>

    </div>

    """, unsafe_allow_html=True)

    st.markdown("### ⚙ Analysis")
    preset_name = st.radio("Depth", list(PRESETS), index=1,
                           label_visibility="collapsed")
    st.caption(PRESETS[preset_name]["note"])
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🤖 AI Explanation")
    use_api = st.toggle("Write it in natural language", value=True,
                        help="Turn off to use the built-in template. "
                             "The template needs no internet.")
    provider = st.selectbox(
    "LLM Provider",
    ["openai","anthropic"],
    disabled=not use_api
    )

cfg = PRESETS[preset_name]

st.markdown("""

<div style="
background:linear-gradient(135deg, rgba(17,24,39,.95), rgba(20,30,48,.95));
border:1px solid rgba(59,130,246,.25);
border-radius:28px;
padding:32px;
margin-bottom:30px;
box-shadow:0 0 35px rgba(34,211,238,.08);
">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
">

<div>

<div style="
font-size:14px;
letter-spacing:3px;
font-weight:700;
color:#38BDF8;
">

UPLOAD FOOTAGE

</div>

<div style="
font-family:'Space Grotesk';
font-size:36px;
font-weight:700;
margin-top:10px;
color:white;
">

Drop CCTV or Dashcam Video

</div>

<div style="
margin-top:10px;
font-size:17px;
line-height:1.7;
color:#94A3B8;
max-width:700px;
">

Supported formats:
MP4 • AVI • MOV • MKV

Maximum size:
200 MB

</div>

</div>

<div style="
font-size:70px;
">

📹

</div>

</div>

</div>

""", unsafe_allow_html=True)

upload = st.file_uploader(
"",
type=["mp4","avi","mov","mkv"],
label_visibility="collapsed"
) 
if upload is None:

    st.markdown("""
<div style="
background:#111827;
border:1px solid rgba(255,255,255,.08);
border-radius:24px;
padding:80px;
text-align:center;
">

<div style="font-size:70px;">
📂
</div>

<div style="
font-family:'Space Grotesk';
font-size:34px;
font-weight:700;
margin-top:20px;
color:white;
">
Waiting for Video
</div>

<div style="
margin-top:15px;
font-size:18px;
line-height:1.8;
color:#94A3B8;
">
Upload surveillance footage to begin AI analysis.<br><br>
Vehicle tracking, collision detection, motion analysis,
timeline reconstruction and report generation
will begin automatically.
</div>

</div>
""", unsafe_allow_html=True)

    st.stop()

tmp = Path(tempfile.gettempdir()) / f"drishtai_{upload.name}"
tmp.write_bytes(upload.getbuffer())
video_path = str(tmp)

try:
    meta = probe(video_path)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

st.markdown("### 📊 Video Overview")

a,b,c,d = st.columns(4)

with a:
    st.metric(
        "🎥 Duration",
        f"{meta['duration']:.1f}s"
    )

with b:
    st.metric(
        "⚡ Frame Rate",
        f"{meta['fps']:.0f} FPS"
    )

with c:
    st.metric(
        "🖼 Total Frames",
        f"{meta['frames']:,}"
    )

with d:
    st.metric(
        "🧠 Frames Analysed",
        f"{meta['frames']//cfg['interval']:,}"
    )

left,right = st.columns([2,1])

with left:

    st.markdown("### 🎬 Uploaded Video")

    st.video(video_path)

with right:

    st.markdown("### ⚙ Analysis Mode")

    # Bar fill and caption both follow the selected preset. Previously both
    # were hardcoded to "Balanced" and 70%, so the card claimed the wrong
    # profile whenever Quick look or Thorough was chosen — the one place in
    # the UI that stated something the pipeline was not doing.
    profile_pct = {"Quick look": 40, "Balanced": 70, "Thorough": 100}[preset_name]

    st.markdown(f"""
<div class="card">

<div style="
font-size:13px;
letter-spacing:2px;
color:#38BDF8;
font-weight:700;
margin-bottom:18px;
">
CURRENT CONFIGURATION
</div>

<div style="
display:inline-block;
padding:8px 16px;
background:rgba(34,197,94,.15);
border:1px solid rgba(34,197,94,.30);
border-radius:999px;
color:#22C55E;
font-weight:700;
margin-bottom:25px;
">
🟢 {preset_name.upper()}
</div>

<div style="
display:flex;
justify-content:space-between;
padding:14px 0;
border-bottom:1px solid rgba(255,255,255,.08);
">
<span style="color:#94A3B8;">🧠 Model</span>
<b>{cfg["model"]}</b>
</div>

<div style="
display:flex;
justify-content:space-between;
padding:14px 0;
border-bottom:1px solid rgba(255,255,255,.08);
">
<span style="color:#94A3B8;">🎞 Interval</span>
<b>Every {cfg["interval"]} Frames</b>
</div>

<div style="
display:flex;
justify-content:space-between;
padding:14px 0;
border-bottom:1px solid rgba(255,255,255,.08);
">
<span style="color:#94A3B8;">📏 Image Size</span>
<b>{cfg["imgsz"]} px</b>
</div>

<div style="
display:flex;
justify-content:space-between;
padding:14px 0;
">
<span style="color:#94A3B8;">⚡ Status</span>

<div style="
padding:6px 12px;
background:rgba(34,197,94,.15);
border:1px solid rgba(34,197,94,.30);
border-radius:999px;
color:#22C55E;
font-weight:700;
">
READY
</div>

</div>

<div style="margin-top:22px;">

<div style="
height:8px;
background:#1E293B;
border-radius:999px;
overflow:hidden;
">

<div style="
height:100%;
width:{profile_pct}%;
background:linear-gradient(90deg,#3B82F6,#22D3EE);
">
</div>

</div>

<div style="
margin-top:10px;
font-size:13px;
color:#94A3B8;
">
{preset_name} performance profile
</div>

</div>

</div>
""", unsafe_allow_html=True)

already_analysed = (
    "result" in st.session_state and
    st.session_state.get("video_path") == video_path
)

busy = st.session_state.get("busy", False)

button_text = (
    "🔄 Run Again" if already_analysed
    else ("⏳ Analysing..." if busy else "🚀 Start AI Investigation")
)

clicked = st.button(
    button_text,
    type="primary",
    use_container_width=True,
    disabled=busy,
    key="analysis_button"
)

# Reserve space BELOW the button
progress = st.empty()
status = st.empty()

st.markdown(
    "<div style='height:240px'></div>",
    unsafe_allow_html=True
)

if clicked:

    if already_analysed:
        st.session_state.pop("result", None)
        st.session_state.pop("elapsed", None)
        st.session_state.pop("video_path", None)
        st.session_state.pop("explanation", None)
        st.session_state.pop("explain_live", None)
        st.rerun()

    st.session_state.busy = True

    STAGES = 4

    def ui(stage, label, frac, detail):

        percent = int(min(((stage - 1) + frac) / STAGES * 100, 100))

        progress.markdown(f"""
<div class="card" style="margin-top:20px;">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
margin-bottom:18px;
">

<div>

<div style="
font-family:'Space Grotesk';
font-size:24px;
font-weight:700;
color:white;
">

{label}

</div>

<div style="
margin-top:5px;
font-size:15px;
color:#94A3B8;
">

{detail}

</div>

</div>

<div style="
font-size:34px;
font-weight:700;
color:#22D3EE;
">

{percent}%

</div>

</div>

<div style="
height:16px;
background:#1E293B;
border-radius:999px;
overflow:hidden;
">

<div style="
height:100%;
width:{percent}%;
background:linear-gradient(90deg,#2563EB,#22D3EE);
transition:width .35s;
">

</div>

</div>

</div>
""", unsafe_allow_html=True)

    t0 = time.time()

    try:
        st.session_state.pop("explanation", None)
        st.session_state.pop("explain_live", None)

        st.session_state.result = analyse(
            video_path,
            cfg,
            ui
        )

        st.session_state.elapsed = time.time() - t0
        st.session_state.video_path = video_path

    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"Analysis stopped: {type(e).__name__} — {e}")
        st.session_state.busy = False
        st.stop()

    ui(
        4,
        "✅ Analysis Complete",
        1.0,
        "Preparing investigation dashboard..."
    )

    time.sleep(1)

    progress.empty()
    status.empty()

    st.session_state.busy = False

    st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

res = st.session_state.get("result")

if not res:
    st.stop()

if res["reason"] == "no_vehicles":
    st.warning("**No vehicles were found in this footage.** The clip may not "
               "show road traffic, or the vehicles may be too small in frame. "
               "Try the Thorough setting, or a clip where vehicles are larger.")
    st.stop()

if res["reason"] == "no_collision":
    st.info("**No collision was detected in this footage.** Vehicles were "
            "tracked successfully, but no pair converged and lost speed in "
            "the way an impact produces.")
    if res["ranked"]:
        st.markdown('<div class="eyebrow">Closest interactions</div>',
                    unsafe_allow_html=True)
        st.dataframe([{"Vehicles": f"{p.a} + {p.b}",
                       "Closest approach": f"{p.min_gap:.2f}",
                       "At frame": p.min_gap_frame,
                       "Closing frames": p.approach_frames,
                       "Speed change": f"{p.velocity_drop:.0%}"}
                      for p in res["ranked"]], use_container_width=True)
    st.stop()

if res:

    st.markdown("""
        ## 📋 Investigation Summary
        """)

    s1, s2, s3 = st.columns(3)

    with s1:
        st.success("✔ Video Processed")

    with s2:
        st.info(f"🎥 {meta['frames']:,} Frames")

    with s3:
        st.info(f"⏱ {st.session_state.get('elapsed',0):.0f} sec")


# ---- Collision found -------------------------------------------------------
tl = res["timeline"]
warning, anchor, lead = res["warning"], res["anchor"], res["lead"]
impact_s = seconds_of(anchor)
warn_s = seconds_of(warning) if warning else None
vehicles = anchor["objects_involved"]

# .get() not [] on purpose: a result cached in session_state from a run
# before this feature existed has no class_map key, and a KeyError here
# would blank the whole dashboard on the first rerun after an update.
# Missing map degrades to bare ids, which is the pre-existing behaviour.
CLASS_MAP = res.get("class_map") or {}

# The anchor is a confirmed collision only when the detector emitted one.
# Otherwise it is the most serious event found, and the headline must say so
# — announcing "Collision found" for an unconfirmed contact would be the one
# claim in this interface that a reviewer could prove wrong.
confirmed = any(r["event"] == "collision" for r in tl)
EVENT_TITLE = {"collision": "Collision found at",
               "sudden_velocity_change": "Sharp speed change at",
               "trajectory_intersecting": "Paths converged at",
               "distance_dropping": "Vehicles closed on each other at"}
headline_label = EVENT_TITLE.get(anchor["event"], "Event found at")

st.markdown("""
<div style="margin-top:25px;"></div>
""", unsafe_allow_html=True)

#############################################
# DASHBOARD CARDS
#############################################

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(f"""
<div class="card">

<div style="color:#94A3B8;font-size:13px;">
COLLISION TIME
</div>

<div style="
font-size:38px;
font-family:'Space Grotesk';
font-weight:700;
margin-top:10px;
">

{humanise_seconds(impact_s)}

</div>

<div style="margin-top:10px;color:#CBD5E1;">
{anchor["timestamp"]}
</div>

</div>
""", unsafe_allow_html=True)

with c2:

    st.markdown(f"""
<div class="card">

<div style="color:#94A3B8;font-size:13px;">
EARLIEST WARNING
</div>

<div style="
font-size:38px;
font-family:'Space Grotesk';
font-weight:700;
color:#F59E0B;
margin-top:10px;
">

{"%.2fs" % lead if lead else "--"}

</div>

<div style="margin-top:10px;color:#CBD5E1;">
Lead Time
</div>

</div>
""", unsafe_allow_html=True)

with c3:

    st.markdown(f"""
<div class="card">

<div style="color:#94A3B8;font-size:13px;">
VEHICLES
</div>

<div style="
font-size:30px;
font-family:'Space Grotesk';
font-weight:700;
margin-top:10px;
">

{"<br>".join(label_for(v, CLASS_MAP) for v in vehicles)}

</div>

</div>
""", unsafe_allow_html=True)

with c4:

    status = "Collision" if confirmed else "Risk Event"

    color = "#EF4444" if confirmed else "#F59E0B"

    st.markdown(f"""
<div class="card">

<div style="color:#94A3B8;font-size:13px;">
STATUS
</div>

<div style="
font-size:34px;
font-family:'Space Grotesk';
font-weight:700;
margin-top:10px;
color:{color};
">

{status}

</div>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("## 🛰 Investigation Timeline")

st.markdown(
    time_ruler(
        res["meta"]["duration"],
        warn_s,
        impact_s,
        "impact" if confirmed else "event"
    ),
    unsafe_allow_html=True
)

left, right = st.columns([1.25, .75])

with left:
    st.markdown("## 📷 Evidence")
    frame = grab_frame(st.session_state.video_path, anchor["frame_index"])
    if frame is not None:
        at_impact = [r for r in res["motion"]
                     if r["frame_index"] == anchor["frame_index"]]
        st.image(draw_boxes(frame, at_impact, vehicles))
        st.markdown(f"""
        <div style="
        text-align:center;
        font-size:24px;
        font-weight:700;
        margin-top:8px;
        margin-bottom:25px;
        color:#E2E8F0;
        ">
        🚨 Collision Frame | 🎞 Frame {anchor['frame_index']} | ⏱ {anchor['timestamp']}
        </div>
        """, unsafe_allow_html=True)
    if warn_s is not None:
        wf = grab_frame(st.session_state.video_path, warning["frame_index"])
        if wf is not None:
            at_warn = [r for r in res["motion"]
                       if r["frame_index"] == warning["frame_index"]]
            st.image(draw_boxes(wf, at_warn, vehicles))
            st.markdown(f"""
            <div style="
            text-align:center;
            font-size:24px;
            font-weight:700;
            margin-top:8px;
            color:#E2E8F0;
            ">
            ⚠ Earliest Warning | 🎞 Frame {warning['frame_index']} | ⏱ Lead Time: {lead:.2f}s
            </div>
            """, unsafe_allow_html=True)

with right:
    st.markdown("## 🤖 AI Investigation Report")
    if "explanation" not in st.session_state:
        with st.spinner("Writing the summary…"):
            if use_api:
                txt, live = explain_with_api(tl, provider=provider)
            else:
                txt, live = explain_offline(tl), False
        st.session_state.explanation = txt
        st.session_state.explain_live = live
    st.markdown(f"""

        <div class="card">

        <div style="
        font-size:15px;
        letter-spacing:2px;
        color:#38BDF8;
        font-weight:700;
        margin-bottom:15px;
        ">

        AI INVESTIGATION REPORT

        </div>

        <div class="explain">

        {humanise_ids(st.session_state.explanation, CLASS_MAP)}

        </div>

        </div>

        """, unsafe_allow_html=True)
    # if not st.session_state.get("explain_live"):
    #     st.caption("Written by the built-in template (no internet needed).")

    st.markdown("### Event Timeline")
    LABEL = {"moving_normally": "Moving normally",
             "distance_dropping": "Gap between them closing",
             "trajectory_intersecting": "Paths converging",
             "sudden_velocity_change": "Sharp change in speed",
             "collision": "Collision"}
    ICON = {

    "moving_normally":"🟢",

    "distance_dropping":"🟡",

    "trajectory_intersecting":"🟠",

    "sudden_velocity_change":"🔴",

    "collision":"💥"

    }

    for r in tl:

        st.markdown(f"""
    <div style="
    background:#111827;
    border-left:4px solid #3B82F6;
    padding:18px;
    margin-bottom:14px;
    border-radius:16px;
    ">

    <div style="
    display:flex;
    justify-content:space-between;
    align-items:center;
    ">

    <div>

    <div style="font-size:20px;">
    {ICON.get(r["event"], "📍")} <strong>{LABEL.get(r["event"], r["event"])}</strong>
    </div>

    <div style="color:#94A3B8;">
    {", ".join(label_for(o, CLASS_MAP) for o in r["objects_involved"])}
    </div>

    </div>

    <div style="
    font-family:monospace;
    font-size:18px;
    ">
    {seconds_of(r):.2f}s
    </div>

    </div>

    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="height:50px;"></div>
""", unsafe_allow_html=True)
# st.markdown("""
# <div class="card">

# <b style="
# font-family:'Space Grotesk';
# font-size:24px;
# color:white;
# ">

# Ask the Investigation AI

# </b>

# <div style="
# margin-top:10px;
# color:#94A3B8;
# ">

# Examples:

# • What caused the collision?

# • Which vehicle was at fault?

# • When was the earliest warning?

# • Explain the timeline.

# </div>

# </div>

# """, unsafe_allow_html=True)
st.markdown("""
<div class="card" style="padding:22px 28px;margin-bottom:18px;">

<div style="
display:flex;
justify-content:space-between;
align-items:center;
">

<div>

<div style="
font-family:'Space Grotesk';
font-size:34px;
font-weight:700;
color:white;
">

🤖 AI Investigation Assistant

</div>

<div style="
margin-top:8px;
font-size:16px;
color:#94A3B8;
">

Ask questions about this incident using the reconstructed timeline.

</div>

</div>

<div style="
padding:10px 18px;
background:rgba(34,197,94,.12);
border:1px solid rgba(34,197,94,.35);
border-radius:999px;
color:#22C55E;
font-weight:700;
">

🟢 ONLINE

</div>

</div>

</div>
""", unsafe_allow_html=True)
q = st.text_input(
"",
placeholder="💬 Ask anything about this investigation...",
label_visibility="collapsed",
key="chatbox"
)

# Suggested questions: a blank text box gives no hint that "why" is even
# answerable. These are the three questions a reviewer actually asks.
st.markdown("""
<div style="
margin-top:10px;
margin-bottom:10px;
font-size:14px;
letter-spacing:2px;
color:#94A3B8;
font-weight:700;
">

SUGGESTED QUESTIONS

</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)

questions = [

"Why did this happen?",

"When did the accident happen?",

"What was the earliest warning?",

"Who slowed first?",

"Explain the timeline"

]

for col,text in zip([c1,c2,c3,c4,c5],questions):

    if col.button(text,use_container_width=True):

        q=text

if q:

    answer = humanise_ids(answer_query(tl, q, motion=res["motion"]), CLASS_MAP)

    st.markdown(f"""
    <div class="card" style="margin-top:25px;">

    <div style="
    display:flex;
    align-items:center;
    gap:16px;
    margin-bottom:25px;
    ">

    <div style="width:54px;
    height:54px;
    border-radius:50%;
    background:linear-gradient(135deg,#2563EB,#22D3EE);
    display:flex;
    justify-content:center;
    align-items:center;
    font-size:24px;">

    🤖

    </div>

    <div>

    <div style="
    font-family:'Space Grotesk';
    font-size:24px;
    font-weight:700;
    ">

    DrishtAI

    </div>

    <div style="
    font-size:14px;
    color:#94A3B8;
    ">

    AI Investigation Assistant

    </div>

    </div>

    </div>

    <div style="
    font-size:18px;
    line-height:1.9;
    color:#E5E7EB;
    ">

    {answer}

    </div>

    </div>
    """,unsafe_allow_html=True)

st.markdown("""
<div style="
margin:50px 0 30px 0;
height:1px;
background:rgba(255,255,255,.08);
"></div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="
display:flex;
align-items:center;
gap:14px;
margin-bottom:18px;
">

<div style="font-size:38px;">
🚗
</div>

<div>

<div style="
font-family:'Space Grotesk';
font-size:34px;
font-weight:700;
color:white;
">

Vehicle Motion Analysis

</div>

<div style="
font-size:15px;
color:#94A3B8;
">

AI explanation of vehicle behaviour before impact

</div>

</div>

</div>
""", unsafe_allow_html=True)
# `vehicles` goes in as raw ids — explain_cause matches them against
# motion records. Only its returned prose is relabelled.
st.markdown(f'<div class="card explain">'
            f'{humanise_ids(explain_cause(res["motion"], vehicles, anchor["frame_index"]), CLASS_MAP)}'
            f'</div>', unsafe_allow_html=True)

with st.expander("⚙ Advanced Technical Details"):
    st.write(f"Processed in {st.session_state.get('elapsed', 0):.0f}s · "
             f"{preset_name} · {cfg['model']} · every "
             f"{cfg['interval']} frame(s)")
    if res.get("merged"):
        st.write("Merged split detections (articulated vehicles):", res["merged"])
    st.markdown("**Candidate pairs considered**")
    st.dataframe([{"Vehicles": f"{p.a} + {p.b}", "Score": p.score,
                   "Contact": p.contact, "Closing frames": p.approach_frames,
                   "Speed change": f"{p.velocity_drop:.0%}"}
                  for p in res["ranked"]], use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download timeline (JSON)", json.dumps(tl, indent=2),
                       "timeline.json", "application/json",
                       use_container_width=True)
    d2.download_button("Download full analysis (JSON)",
                       json.dumps({"events": res["events"], "timeline": tl},
                                  indent=2),
                       "analysis.json", "application/json",
                       use_container_width=True)

st.markdown("""
<br><br>

<hr>

<div style="
text-align:center;
padding:30px;
color:#64748B;
font-size:14px;
">

Built with

YOLOv8 • ByteTrack • OpenCV • Streamlit • OpenAI

<br><br>

© 2026 DrishtAI

</div>

""", unsafe_allow_html=True)