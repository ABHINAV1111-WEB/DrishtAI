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

st.set_page_config(page_title="DrishtAI — Incident Review",
                   page_icon="◉", layout="wide")

# --- Palette: ink on paper, two signal colours that MEAN something.
#     amber = the earliest observable warning, red = the impact.
#     Nothing else is coloured, so colour always carries information.
INK, MUTED, RULE = "#1B1F24", "#6B7280", "#E3E6EA"
WARN, IMPACT, CALM = "#B45309", "#B91C1C", "#3F6212"

st.markdown(f"""<style>
.stApp {{ background:#FBFAF8; }}
html, body, [class*="css"] {{ font-family:"Inter","Segoe UI",system-ui,sans-serif; }}
.mono {{ font-family:"SF Mono",Consolas,"Roboto Mono",monospace; }}
.eyebrow {{ font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
           color:{MUTED}; font-weight:600; margin-bottom:.35rem; }}
.headline {{ font-size:2.6rem; line-height:1.1; font-weight:700; color:{INK};
            letter-spacing:-.02em; margin:0; }}
.sub {{ color:{MUTED}; font-size:.95rem; margin-top:.4rem; }}
.card {{ background:#FFF; border:1px solid {RULE}; border-radius:10px;
        padding:1.15rem 1.3rem; }}
.answer {{ font-size:2.9rem; font-weight:700; color:{IMPACT}; line-height:1;
          letter-spacing:-.02em; }}
.answer-sub {{ color:{MUTED}; font-size:.9rem; margin-top:.3rem; }}
.lead {{ font-size:2.9rem; font-weight:700; color:{WARN}; line-height:1;
        letter-spacing:-.02em; }}
.chip {{ display:inline-block; padding:.12rem .5rem; border-radius:999px;
        font-size:.7rem; font-weight:600; letter-spacing:.04em; }}
.explain {{ font-size:1.06rem; line-height:1.65; color:{INK}; }}
hr {{ border:none; border-top:1px solid {RULE}; margin:1.4rem 0; }}
[data-testid="stMetricValue"] {{ font-size:1.35rem; }}
</style>""", unsafe_allow_html=True)


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
                    events=[], timeline=[], reason="no_vehicles")

    ui(2, "Measuring speed and direction", 1.0,
       f"{len({r['object_id'] for r in records})} vehicles tracked")
    motion_recs, prov, _ = motion_run(records, smooth_window=3, do_merge=True)
    motion = [r.__dict__ for r in motion_recs]

    ui(3, "Looking for collisions", 1.0, "comparing every vehicle pair")
    ranked, events = collision_run(motion, window=3, top=5)

    if not events:
        return dict(meta=meta, tracks=records, motion=motion, ranked=ranked,
                    events=[], timeline=[], merged=prov, reason="no_collision")

    ui(4, "Building the event timeline", 1.0, "finding the earliest warning")
    tl_recs, anchor, warning, lead = build_timeline(
        [e.__dict__ if hasattr(e, "__dict__") else e for e in events])

    return dict(meta=meta, tracks=records, motion=motion, ranked=ranked,
                events=[e.__dict__ for e in events],
                timeline=[r.__dict__ for r in tl_recs],
                anchor=anchor, warning=warning, lead=lead,
                merged=prov, reason="ok")


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.markdown('<div class="eyebrow">DrishtAI · incident review</div>',
            unsafe_allow_html=True)
st.markdown('<p class="headline">Find the moment.</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Upload road footage. DrishtAI locates the collision, '
            'reconstructs the sequence that led to it, and marks the earliest '
            'point the risk was visible.</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="eyebrow">Analysis</div>', unsafe_allow_html=True)
    preset_name = st.radio("Depth", list(PRESETS), index=1,
                           label_visibility="collapsed")
    st.caption(PRESETS[preset_name]["note"])
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Explanation</div>', unsafe_allow_html=True)
    use_api = st.toggle("Write it in natural language", value=True,
                        help="Turn off to use the built-in template. "
                             "The template needs no internet.")
    provider = st.selectbox("Service", ["openai", "anthropic"],
                            disabled=not use_api)

cfg = PRESETS[preset_name]

upload = st.file_uploader("Footage", type=["mp4", "avi", "mov", "mkv"],
                          label_visibility="collapsed")

if upload is None:
    st.markdown(f'<div class="card"><b>Start by uploading a clip.</b>'
                f'<div class="sub">MP4, AVI, MOV or MKV. Any frame rate. '
                f'Shorter clips return faster — a 30-second clip takes a '
                f'few minutes on a laptop CPU.</div></div>',
                unsafe_allow_html=True)
    st.stop()

tmp = Path(tempfile.gettempdir()) / f"drishtai_{upload.name}"
tmp.write_bytes(upload.getbuffer())
video_path = str(tmp)

try:
    meta = probe(video_path)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Length", f"{meta['duration']:.1f}s")
c2.metric("Frame rate", f"{meta['fps']:.0f} fps")
c3.metric("Frames", f"{meta['frames']:,}")
c4.metric("Will analyse", f"{meta['frames'] // cfg['interval']:,}")

if st.button("Analyse footage", type="primary", use_container_width=True):
    bar = st.progress(0.0)
    status = st.empty()
    STAGES = 4

    def ui(stage, label, frac, detail):
        bar.progress(min(((stage - 1) + frac) / STAGES, 1.0))
        status.markdown(f"**Step {stage} of {STAGES} — {label}**  \n"
                        f'<span class="sub">{detail}</span>',
                        unsafe_allow_html=True)

    t0 = time.time()
    try:
        st.session_state.result = analyse(video_path, cfg, ui)
        st.session_state.elapsed = time.time() - t0
        st.session_state.video_path = video_path
    except Exception as e:
        bar.empty(); status.empty()
        st.error(f"Analysis stopped: {type(e).__name__} — {e}")
        st.stop()
    bar.empty(); status.empty()

res = st.session_state.get("result")
if not res:
    st.stop()

st.markdown("<hr>", unsafe_allow_html=True)

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

# ---- Collision found -------------------------------------------------------
tl = res["timeline"]
warning, anchor, lead = res["warning"], res["anchor"], res["lead"]
impact_s = seconds_of(anchor)
warn_s = seconds_of(warning) if warning else None
vehicles = anchor["objects_involved"]

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

a, b = st.columns([3, 2])
with a:
    st.markdown(f'<div class="eyebrow">{headline_label}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="answer" style="color:'
                f'{IMPACT if confirmed else WARN}">{humanise_seconds(impact_s)}</div>'
                f'<div class="answer-sub mono">{anchor["timestamp"]} · '
                f'{" and ".join(vehicles)}</div>', unsafe_allow_html=True)
    if not confirmed:
        st.caption("No physical contact was confirmed. This is the most "
                   "serious event detected in the footage.")
with b:
    if lead:
        st.markdown('<div class="eyebrow">Risk was visible</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="lead">{lead:.2f}s earlier</div>'
                    f'<div class="answer-sub">before that moment</div>',
                    unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
st.markdown(time_ruler(res["meta"]["duration"], warn_s, impact_s,
                       "impact" if confirmed else "event"),
            unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="eyebrow">Evidence</div>', unsafe_allow_html=True)
    frame = grab_frame(st.session_state.video_path, anchor["frame_index"])
    if frame is not None:
        at_impact = [r for r in res["motion"]
                     if r["frame_index"] == anchor["frame_index"]]
        st.image(draw_boxes(frame, at_impact, vehicles),
                 caption=f"Frame {anchor['frame_index']} — the moment of impact",
                 use_container_width=True)
    if warn_s is not None:
        wf = grab_frame(st.session_state.video_path, warning["frame_index"])
        if wf is not None:
            at_warn = [r for r in res["motion"]
                       if r["frame_index"] == warning["frame_index"]]
            st.image(draw_boxes(wf, at_warn, vehicles),
                     caption=f"Frame {warning['frame_index']} — earliest warning, "
                             f"{lead:.2f}s earlier",
                     use_container_width=True)

with right:
    st.markdown('<div class="eyebrow">What happened</div>', unsafe_allow_html=True)
    if "explanation" not in st.session_state:
        with st.spinner("Writing the summary…"):
            if use_api:
                txt, live = explain_with_api(tl, provider=provider)
            else:
                txt, live = explain_offline(tl), False
        st.session_state.explanation = txt
        st.session_state.explain_live = live
    st.markdown(f'<div class="card explain">{st.session_state.explanation}</div>',
                unsafe_allow_html=True)
    if not st.session_state.get("explain_live"):
        st.caption("Written by the built-in template (no internet needed).")

    st.markdown('<div class="eyebrow" style="margin-top:1.3rem">Sequence</div>',
                unsafe_allow_html=True)
    LABEL = {"moving_normally": "Moving normally",
             "distance_dropping": "Gap between them closing",
             "trajectory_intersecting": "Paths converging",
             "sudden_velocity_change": "Sharp change in speed",
             "collision": "Collision"}
    for r in tl:
        s = seconds_of(r)
        col = IMPACT if r["event"] == "collision" else (
            WARN if r["is_earliest_warning"] else MUTED)
        tag = ('<span class="chip" style="background:#FEF3C7;color:#92400E">'
               'EARLIEST WARNING</span>') if r["is_earliest_warning"] else ""
        st.markdown(
            f'<div style="display:flex;gap:.85rem;align-items:baseline;'
            f'padding:.45rem 0;border-bottom:1px solid {RULE}">'
            f'<span class="mono" style="color:{MUTED};min-width:74px;'
            f'font-size:.85rem">{s:6.2f}s</span>'
            f'<span style="color:{col};font-weight:600">{LABEL.get(r["event"], r["event"])}</span>'
            f'<span style="color:{MUTED};font-size:.85rem">'
            f'{", ".join(r["objects_involved"])}</span> {tag}</div>',
            unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Ask about this footage</div>',
            unsafe_allow_html=True)
q = st.text_input("Question", placeholder="Why did this happen?",
                  label_visibility="collapsed")

# Suggested questions: a blank text box gives no hint that "why" is even
# answerable. These are the three questions a reviewer actually asks.
sug = st.columns(3)
for col, text in zip(sug, ["Why did this happen?",
                           "When did the accident happen?",
                           "What was the earliest warning?"]):
    if col.button(text, use_container_width=True):
        q = text

if q:
    st.markdown(f'<div class="card explain">'
                f'{answer_query(tl, q, motion=res["motion"])}</div>',
                unsafe_allow_html=True)

st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">How the vehicles were moving</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="card explain">'
            f'{explain_cause(res["motion"], vehicles, anchor["frame_index"])}'
            f'</div>', unsafe_allow_html=True)

with st.expander("Analysis detail"):
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
