"""
explain.py — DrishtAI pipeline, Stage 7 (plain-language explanation)

Turns timeline.json into the sentences a human actually reads: what
happened, when, and what the earliest observable warning was.

=============================================================================
DEMO SAFETY: THIS MODULE MUST NEVER CRASH THE DEMO
=============================================================================
The explanation is the last thing in the pipeline and the first thing a
judge sees. It is also the only stage that depends on a network call, an
API key, and a third-party service being up — on conference wifi.

So every failure path falls back to `explain_offline()`, a deterministic
template renderer that produces a correct (if plainer) explanation with no
network at all. A missing key, a timeout, a rate limit, a malformed
response: all degrade to the offline text rather than raising.

Run with --offline to force it. Rehearse with --offline at least once so
you know exactly what the fallback looks like on screen.
=============================================================================

Design decisions:

1. THE TIMELINE IS PASSED AS DATA, NOT PROSE. The prompt embeds the actual
   JSON. The model's job is to narrate a structure we computed, not to
   infer events from a description — that keeps the reasoning in our
   pipeline (where it is auditable) and uses the model for language only.
   This matters for the "AI-First" scoring criterion: the model is doing
   what models are good at, not papering over missing logic.

2. THE PROMPT FORBIDS INVENTION EXPLICITLY. The model is told to describe
   only what is in the timeline, to use the given timestamps verbatim, and
   to say nothing about cause, fault, injury or intent — none of which our
   pipeline can observe. A judge asking "how do you stop it hallucinating?"
   gets a concrete answer.

3. API KEY COMES FROM THE ENVIRONMENT, NEVER FROM CODE OR ARGS. A key in a
   source file ends up in git history — and once pushed, it is compromised
   even if you delete it in a later commit. Set OPENAI_API_KEY (or
   ANTHROPIC_API_KEY) in your shell.

3b. TWO PROVIDERS ARE SUPPORTED. OpenAI is the default because IIT Jammu
   supplies a free key; Anthropic is available with --provider anthropic,
   which keeps the pitch deck's stated stack honest. They differ only in
   where the system prompt goes and how the response is unpacked, so the
   shared failure handling covers both. In practice this also means a
   third fallback level: if one provider is down mid-demo, switch with a
   flag rather than dropping to the offline template.

4. TEMPERATURE 0. The same timeline should produce the same explanation
   every run — a demo that reads differently each rehearsal is a demo you
   cannot rehearse.

5. QUERY HANDLING IS RULE-BASED, NOT A SECOND MODEL CALL. Mapping "when did
   the accident happen?" to a timestamp is a lookup over a timeline we
   already built. Spending an API round-trip (and a failure mode) on it
   would be worse, not more impressive.

Usage:
    setx OPENAI_API_KEY "sk-..."             (Windows, then REOPEN terminal)
    python src/reasoning/explain.py timeline.json --fps 30
    python src/reasoning/explain.py timeline.json --out explanation.txt
    python src/reasoning/explain.py timeline.json --provider anthropic
    python src/reasoning/explain.py timeline.json --offline
    python src/reasoning/explain.py timeline.json --dry-run
    python src/reasoning/explain.py timeline.json --query "when did the accident happen"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Provider defaults. OpenAI is the default because the institution supplies
# a free key; Anthropic remains supported so the pitch deck's stated stack
# ("Claude API for the explanation layer") is still true and usable.
DEFAULT_PROVIDER = "openai"
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
}
ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

SYSTEM_PROMPT = """You are the explanation layer of DrishtAI, a CCTV accident-analysis system.

You will be given a JSON timeline of events that a computer-vision pipeline
detected in road footage. Write a short, factual explanation for a human
reviewing the incident.

Rules:
- Describe ONLY events present in the timeline. Invent nothing.
- Each event carries `readable_time`, an already-formatted phrase such as
  "1.83 seconds in". COPY IT VERBATIM. Never compute a time yourself and
  never read a time out of the `timestamp` field: its last component is a
  frame number, not hundredths of a second, so "00:00:01:25" is NOT 1.25
  seconds. Optionally show the raw timestamp in square brackets after the
  readable phrase, e.g. "1.83 seconds in [00:00:01:25]".
- The same event must never be given two different times in your answer.
- Refer to vehicles by their given ids (e.g. vehicle_2).
- The event flagged is_earliest_warning is the first observable moment that
  made the outcome likely. State it explicitly and say how far ahead of the
  collision it occurred.
- Do NOT speculate about cause, fault, blame, injuries, driver intent, or
  anything not observable in the timeline.
- Do not describe the detection method or mention JSON.

Format:
1. One sentence summarising the incident and its timestamp.
2. Two to four sentences walking the sequence in order.
3. One sentence naming the earliest warning and its lead time.

Plain prose. No headings, no bullet points, under 150 words."""

EVENT_PHRASES = {
    "moving_normally": "was moving normally",
    "distance_dropping": "the gap between them began closing",
    "trajectory_intersecting": "their paths were converging",
    "sudden_velocity_change": "there was a sharp change in speed",
    "collision": "they collided",
}


# ---------------------------------------------------------------------------
# Timeline reading
# ---------------------------------------------------------------------------

def timeline_facts(timeline: list[dict]) -> dict:
    """Pull the handful of facts both the prompt and the fallback need."""
    collision = next((r for r in timeline if r["event"] == "collision"), None)
    warning = next((r for r in timeline if r.get("is_earliest_warning")), None)

    # The most severe event actually present. A clip can contain converging
    # vehicles and a sharp speed change without the collision test passing
    # (contact is measured in a 2D projection and is deliberately strict).
    # Downstream answers should describe THAT event rather than either
    # claiming a collision that was never confirmed or refusing to answer.
    SEVERITY = ["moving_normally", "distance_dropping",
                "trajectory_intersecting", "sudden_velocity_change", "collision"]
    ranked = [r for r in timeline if r["event"] in SEVERITY]
    most_severe = max(ranked, key=lambda r: SEVERITY.index(r["event"])) if ranked else None
    vehicles: list[str] = []
    for r in timeline:
        for v in r["objects_involved"]:
            if v not in vehicles:
                vehicles.append(v)
    anchor = collision or most_severe or (timeline[-1] if timeline else None)
    return {
        "collision": collision,
        "most_severe": most_severe,
        "confirmed_collision": collision is not None,
        "anchor": anchor,
        "warning": warning,
        "vehicles": vehicles,
        "n_events": len(timeline),
    }


def seconds_of(record: dict, fps: int | None = None) -> float:
    """
    Seconds for a timeline record.

    Prefers the `time_seconds` float that Stage 6 carries through — it is
    exact, and needs no knowledge of the source frame rate. Only if a
    record predates that change do we fall back to parsing HH:MM:SS:FF,
    which DOES need fps and will refuse rather than guess.
    """
    if "time_seconds" in record:
        return float(record["time_seconds"])
    if fps is None:
        raise ValueError(
            f"Record {record.get('timestamp')!r} has no time_seconds field "
            f"and no --fps was given. Re-run Stage 6 (timeline_builder.py) "
            f"to regenerate timeline.json with time_seconds included, or "
            f"pass --fps <source video fps> to parse the timestamp instead."
        )
    return timestamp_to_seconds(record["timestamp"], fps)


def timestamp_to_seconds(ts: str, fps: int) -> float:
    """
    HH:MM:SS:FF -> float seconds.

    fps is REQUIRED and has no default on purpose. FF is "frames within
    this second", so the same string means different times at different
    frame rates: 00:00:01:25 is 1.42 s at 60 fps but 1.83 s at 30 fps.
    A default here silently produced a lead time of 1.22 s for a clip whose
    true lead time was 1.43 s — no error, just a wrong number in the demo.
    Callers must state the fps of the source video.
    """
    try:
        hh, mm, ss, ff = (int(p) for p in ts.split(":"))
    except ValueError:
        return 0.0
    return hh * 3600 + mm * 60 + ss + ff / fps


def humanise_seconds(total: float) -> str:
    """
    Seconds -> a phrase a person can read out loud.

    The schema's HH:MM:SS:FF is frame-exact and correct for seeking, but
    "00:00:01:25" is a broadcast convention that a viewer cannot decode:
    the last field is frames-within-a-second, not hundredths. Anything
    human-facing gets this instead; the raw timestamp stays in the JSON and
    in the UI's precise-seek field.
    """
    if total < 60:
        return f"{total:.2f} seconds in"
    minutes, seconds = divmod(total, 60)
    if minutes < 60:
        return f"{int(minutes)} min {seconds:.1f} s in"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes} min {seconds:.1f} s in"


def lead_time(facts: dict, fps: int | None = None) -> float | None:
    if not facts["warning"] or not facts["anchor"]:
        return None
    return round(seconds_of(facts["anchor"], fps)
                 - seconds_of(facts["warning"], fps), 2)


# ---------------------------------------------------------------------------
# Offline fallback — deterministic, no network
# ---------------------------------------------------------------------------

def explain_offline(timeline: list[dict], fps: int | None = None) -> str:
    """
    Template explanation. Correct and plain rather than fluent. This is what
    the demo shows if the API is unreachable, so it must always produce
    something sensible — including when there is no collision at all.
    """
    if not timeline:
        return "No events were detected in this footage."

    f = timeline_facts(timeline)
    lead = lead_time(f, fps)
    parts: list[str] = []

    if f["collision"]:
        vs = " and ".join(f["collision"]["objects_involved"])
        parts.append(f"A collision between {vs} was detected "
                     f"{humanise_seconds(seconds_of(f['collision'], fps))} "
                     f"({f['collision']['timestamp']}).")
    else:
        parts.append("No collision was confirmed in this footage, but "
                     "risk-elevating behaviour was detected.")

    seen: set[tuple[str, str]] = set()
    for r in timeline:
        key = (r["timestamp"], r["event"])
        if r["event"] == "moving_normally" or key in seen:
            continue
        seen.add(key)
        phrase = EVENT_PHRASES.get(r["event"], r["event"].replace("_", " "))
        vs = " and ".join(r["objects_involved"])
        parts.append(f"At {humanise_seconds(seconds_of(r, fps))}, {phrase} ({vs}).")

    if f["warning"]:
        vs = " and ".join(f["warning"]["objects_involved"])
        lead_txt = f" — {lead:.2f} seconds before the collision" if lead else ""
        parts.append(f"The earliest observable warning came "
                     f"{humanise_seconds(seconds_of(f['warning'], fps))}{lead_txt}, when "
                     f"{EVENT_PHRASES.get(f['warning']['event'], f['warning']['event'])} "
                     f"({vs}).")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------

def build_prompt(timeline: list[dict], fps: int | None = None) -> str:
    f = timeline_facts(timeline)
    lead = lead_time(f, fps)
    lead_line = (f"Lead time from earliest warning to collision: {lead:.2f} seconds."
                 if lead is not None else
                 "Lead time could not be computed for this clip.")
    # Hand the model FINISHED human-readable times rather than asking it
    # to convert. Given only "00:00:01:25" and time_seconds, GPT-4o-mini
    # read the timestamp as "1.25 seconds" in one sentence while correctly
    # using 1.83 from time_seconds in another — self-contradicting output.
    # The last field of HH:MM:SS:FF is frames, not hundredths, and no
    # amount of prompt wording reliably beats that visual similarity. So we
    # render the phrase in Python (where it is exact) and the model only
    # copies it. Same principle as carrying time_seconds through Stage 6:
    # never make a consumer re-derive a value we already hold.
    annotated = []
    for r in timeline:
        rec = dict(r)
        rec["readable_time"] = humanise_seconds(seconds_of(r, fps))
        annotated.append(rec)

    return (
        "Here is the detected event timeline:\n\n"
        + json.dumps(annotated, indent=2)
        + f"\n\n{lead_line}\n\nWrite the explanation."
    )


def _call_openai(prompt: str, model: str, api_key: str) -> str:
    """OpenAI chat-completions call. System prompt goes in the messages list."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,                 # design note 4: reproducible demos
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _call_anthropic(prompt: str, model: str, api_key: str) -> str:
    """Anthropic call. System prompt is a separate top-level parameter."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=400,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


CALLERS = {"openai": _call_openai, "anthropic": _call_anthropic}


def explain_with_api(
    timeline: list[dict],
    fps: int | None = None,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
) -> tuple[str, bool]:
    """
    Returns (text, used_api). Falls back to the offline template on ANY
    failure — see the demo-safety note at the top of this file.

    The two providers differ in exactly two ways: where the system prompt
    goes, and how the response text is unpacked. Both are isolated in the
    _call_* helpers above so the failure handling below is shared.
    """
    if provider not in CALLERS:
        print(f"[explain] unknown provider {provider!r} — using offline explanation",
              file=sys.stderr)
        return explain_offline(timeline, fps), False

    model = model or DEFAULT_MODELS[provider]
    env_var = ENV_KEYS[provider]
    api_key = os.environ.get(env_var)

    if not api_key:
        print(f"[explain] {env_var} not set — using offline explanation",
              file=sys.stderr)
        return explain_offline(timeline, fps), False

    try:
        text = CALLERS[provider](build_prompt(timeline, fps), model, api_key)
        if not text:
            raise ValueError("empty response from model")
        return text, True
    except ImportError:
        pkg = "openai" if provider == "openai" else "anthropic"
        print(f"[explain] {pkg} package not installed (pip install {pkg}) — "
              f"using offline explanation", file=sys.stderr)
        return explain_offline(timeline, fps), False
    except Exception as e:                      # deliberately broad: see header
        print(f"[explain] {provider} call failed ({type(e).__name__}: {e}) — "
              f"using offline explanation", file=sys.stderr)
        return explain_offline(timeline, fps), False



# ---------------------------------------------------------------------------
# Causal analysis — observable factors, NOT fault
# ---------------------------------------------------------------------------
#
# "Why did this happen?" is the question a reviewer actually asks, and the
# honest answer sits between two failures:
#
#   Refusing entirely ("cannot determine cause") throws away real
#   information. The kinematics DO show who closed the distance, whether
#   anyone braked, and what the impact geometry was.
#
#   Assigning fault ("vehicle_2 was driving carelessly") claims something
#   the data cannot support. Fault needs right of way, signals, lane
#   markings, speed limits, visibility — none of which a bounding box sees.
#   A rear-end collision is usually the following driver's fault, but not
#   if the lead vehicle reversed or ran a light and stopped dead, and our
#   data cannot tell those apart.
#
# So this reports MEASURED ASYMMETRIES and lets the human adjudicate.
# Speeds stay in px/s because without homography there is no real-world
# speed; comparisons between the two vehicles are still valid because both
# are measured in the same units in the same frame.

def _window(motion: list[dict], oid: str, frame: int, before: int, after: int
            ) -> tuple[list[dict], list[dict]]:
    recs = sorted((r for r in motion if r["object_id"] == oid),
                  key=lambda r: r["frame_index"])
    idx = next((i for i, r in enumerate(recs) if r["frame_index"] >= frame), None)
    if idx is None:
        return [], []
    return recs[max(0, idx - before):idx], recs[idx:idx + after + 1]


def causal_factors(motion: list[dict], vehicles: list[str], impact_frame: int,
                   window: int = 5, brake_lookback: int = 18) -> dict:
    """Measurable facts about how the two vehicles were moving at impact."""
    out: dict = {"vehicles": {}, "geometry": None, "closing_vehicle": None}
    speeds: dict[str, float] = {}
    headings: dict[str, float] = {}

    for oid in vehicles:
        before, after = _window(motion, oid, impact_frame, window, window)
        if not before:
            continue
        # Braking needs a LONGER lookback than speed. A driver brakes well
        # before contact, so a 5-frame window sits entirely inside the
        # already-slowed period and sees a flat line. 18 frames (~0.6 s at
        # 30 fps) reaches back to before the brakes were applied.
        brake_win, _ = _window(motion, oid, impact_frame, brake_lookback, 0)
        v_before = sum(r["velocity"] for r in before) / len(before)
        v_after = (sum(r["velocity"] for r in after) / len(after)) if after else v_before
        # Braking = a SUSTAINED speed decline before contact, measured as
        # the mean of the second half of the window against the first half.
        #
        # Counting individual negative acceleration readings does not work.
        # Bounding boxes wobble a few pixels even on a steady vehicle, and
        # differentiating position twice amplifies that into acceleration
        # spikes well past -200 px/s^2. A constant-speed vehicle in testing
        # was reported as "braking" on that rule. Comparing window means
        # cancels the zero-mean jitter and keeps only a real trend — the
        # same reasoning the collision detector uses for impact.
        bw = brake_win or before
        half = max(len(bw) // 2, 1)
        early = sum(r["velocity"] for r in bw[:half]) / half
        late = sum(r["velocity"] for r in bw[half:]) / max(len(bw) - half, 1)
        braked = early > 1.0 and (late - early) / early < -0.15
        speeds[oid] = v_before
        headings[oid] = before[-1]["direction"]
        out["vehicles"][oid] = {
            "speed_before": round(v_before, 1),
            "speed_after": round(v_after, 1),
            "speed_change": round((v_after - v_before) / v_before, 3) if v_before > 1 else 0.0,
            "braked_before_impact": braked,
            "class": before[-1].get("vehicle_class", "vehicle"),
        }

    if len(speeds) == 2:
        fast, slow = sorted(speeds, key=lambda k: speeds[k], reverse=True)
        # Only call it "closing" if the difference is decisive; two vehicles
        # at similar speeds is not an asymmetry worth asserting.
        if speeds[fast] > max(speeds[slow] * 1.5, speeds[slow] + 20):
            out["closing_vehicle"] = fast
            out["slower_vehicle"] = slow

        # Heading is only meaningful for a vehicle that is actually moving.
        # A near-stationary vehicle's centroid is dominated by box wobble,
        # so its "direction" is noise — in testing a rear-end was reported
        # as an 80-degree side impact purely because the slow lead vehicle
        # had a random heading. Below this speed we decline to state the
        # geometry rather than state it wrongly.
        MIN_SPEED_FOR_HEADING = 30.0
        if min(speeds.values()) >= MIN_SPEED_FOR_HEADING:
            diff = abs(headings[fast] - headings[slow]) % 360
            diff = diff if diff <= 180 else 360 - diff
            out["heading_difference"] = round(diff, 1)
            out["geometry"] = ("rear-end" if diff < 35 else
                               "head-on" if diff > 145 else
                               "angled or side impact")
        else:
            slowest = min(speeds, key=lambda k: speeds[k])
            out["geometry_note"] = (
                f"{slowest} was too slow for its heading to be measured "
                f"reliably, so the impact angle is not stated.")

    return out


def explain_cause(motion: list[dict], vehicles: list[str],
                  impact_frame: int) -> str:
    """Plain-English answer to 'why did this happen?'."""
    f = causal_factors(motion, vehicles, impact_frame)
    if not f["vehicles"]:
        return ("There is not enough tracked motion around the impact to "
                "describe how the vehicles were moving.")

    lines: list[str] = []
    closer = f.get("closing_vehicle")

    if closer:
        slower = f["slower_vehicle"]
        cv, sv = f["vehicles"][closer], f["vehicles"][slower]
        slow_desc = ("near-stationary" if sv["speed_before"] < 20
                     else f"slower-moving ({sv['speed_before']:.0f} px/s)")
        lines.append(f"{closer} was closing on a {slow_desc} {slower} at "
                     f"{cv['speed_before']:.0f} px/s.")
        lines.append(f"{closer} "
                     + ("was braking before contact"
                        if cv["braked_before_impact"]
                        else "showed no braking before contact")
                     + ".")
    else:
        parts = [f"{o} at {d['speed_before']:.0f} px/s"
                 for o, d in f["vehicles"].items()]
        lines.append("Both vehicles were moving at comparable speeds ("
                     + " and ".join(parts) + ").")
        braking = [o for o, d in f["vehicles"].items() if d["braked_before_impact"]]
        lines.append(("Braking was detected from " + " and ".join(braking) + "."
                      ) if braking else "Neither vehicle braked before contact.")

    if f.get("geometry"):
        article = "an" if f["geometry"][0] in "aeiou" else "a"
        lines.append(f"The impact geometry is consistent with {article} "
                     f"{f['geometry']} collision "
                     f"(headings differed by {f['heading_difference']:.0f} degrees).")

    elif f.get("geometry_note"):
        lines.append(f["geometry_note"])

    biggest = max(f["vehicles"].items(), key=lambda kv: -kv[1]["speed_change"])
    if biggest[1]["speed_change"] < -0.15:
        lines.append(f"{biggest[0]} lost "
                     f"{abs(biggest[1]['speed_change']):.0%} of its speed at impact.")

    lines.append("These are observations of how the vehicles moved. Deciding "
                 "who was at fault needs right of way, signals, lane markings "
                 "and speed limits, which are not visible to this system.")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Query handling (design note 5)
# ---------------------------------------------------------------------------

QUERY_KEYWORDS = {
    "collision": ["accident", "crash", "collision", "collide", "impact", "hit"],
    "distance_dropping": ["close", "closing", "gap", "approach", "near"],
    "trajectory_intersecting": ["converge", "path", "trajectory", "cross"],
    "sudden_velocity_change": ["brake", "braking", "slow", "speed", "stop"],
}


WHY_WORDS = ["why", "cause", "caused", "reason", "how did", "what went wrong",
             "who was", "fault", "blame", "responsible", "wrongly", "mistake"]


def answer_query(timeline: list[dict], question: str, fps: int | None = None,
                 motion: list[dict] | None = None) -> str:
    """
    Map a natural question to an answer from the timeline.

    `motion` is optional and only needed for "why did this happen" style
    questions, which reason over speeds and headings rather than just
    timestamps.
    """
    if not timeline:
        return "No events were detected in this footage."

    q = question.lower()
    f = timeline_facts(timeline)

    if any(w in q for w in WHY_WORDS):
        # Answer from the ANCHOR, not strictly from a `collision` event.
        #
        # Stage 5 only emits `collision` when contact AND a sustained speed
        # drop are both confirmed. A clip can produce a clear approach and
        # impact-like speed change without clearing that bar, and Stage 6
        # then anchors the timeline on the most severe event it did find.
        # Requiring a literal `collision` here made the UI refuse to explain
        # an incident it was simultaneously displaying evidence for.
        anchor = f["collision"] or f["anchor"]
        if anchor is None:
            return "No events were detected in this footage."
        if motion is None:
            return ("Motion data is needed to answer that. Ask this in the "
                    "app, or pass motion.json to answer_query.")

        text = explain_cause(motion, anchor["objects_involved"],
                             anchor.get("frame_index", 0))
        if not f["collision"]:
            text = ("No contact was confirmed in this footage, so this "
                    "describes the closest interaction rather than an "
                    "impact. " + text)
        return text

    if any(w in q for w in ["warning", "earliest", "first sign", "before"]):
        if f["warning"]:
            lead = lead_time(f, fps)
            extra = f", {lead:.2f} s before the collision" if lead else ""
            return (f"The earliest warning came "
                    f"{humanise_seconds(seconds_of(f['warning'], fps))}{extra}: "
                    f"{f['warning']['event'].replace('_', ' ')} involving "
                    f"{' and '.join(f['warning']['objects_involved'])} "
                    f"[{f['warning']['timestamp']}].")
        return "No earliest warning was identified in this footage."

    for event, words in QUERY_KEYWORDS.items():
        if any(w in q for w in words):
            match = next((r for r in timeline if r["event"] == event), None)
            if match:
                return (f"{event.replace('_', ' ').capitalize()} occurred "
                        f"{humanise_seconds(seconds_of(match, fps))}, involving "
                        f"{' and '.join(match['objects_involved'])} "
                        f"[{match['timestamp']}].")
            return f"No {event.replace('_', ' ')} event was detected in this footage."

    if f["collision"]:
        return (f"The collision was detected "
                f"{humanise_seconds(seconds_of(f['collision'], fps))}, involving "
                f"{' and '.join(f['collision']['objects_involved'])} "
                f"[{f['collision']['timestamp']}].")
    return "No collision was detected in this footage."


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="DrishtAI Stage 7: plain-language explanation")
    p.add_argument("timeline", help="timeline.json from Stage 6")
    p.add_argument("--out", default=None, help="write explanation to this file")
    p.add_argument("--provider", default=DEFAULT_PROVIDER,
                   choices=sorted(CALLERS),
                   help="which API to use (default openai)")
    p.add_argument("--model", default=None,
                   help="model id; defaults to the provider's default "
                        "(openai: gpt-4o-mini, anthropic: claude-sonnet-5). "
                        "Set this if your key does not have access to the "
                        "default model.")
    p.add_argument("--fps", type=int, default=None,
                   help="Normally NOT needed: timeline.json carries exact "
                        "time_seconds, so this works at 30, 60, 120 fps or "
                        "anything else with no flag. Only required for old "
                        "timeline files generated before time_seconds was "
                        "carried through.")
    p.add_argument("--offline", action="store_true",
                   help="force the offline template, no API call")
    p.add_argument("--dry-run", action="store_true",
                   help="print the prompt that would be sent, then stop")
    p.add_argument("--query", default=None,
                   help="ask a question instead of generating an explanation")
    args = p.parse_args()

    src = Path(args.timeline)
    if not src.exists():
        print(f"[explain] ERROR: {src} not found", file=sys.stderr)
        return 1
    timeline = json.loads(src.read_text())

    # A missing time_seconds is a user-fixable situation, not a crash:
    # report it as a clear message rather than a traceback.
    try:
        if args.query:
            print(answer_query(timeline, args.query, args.fps))
            return 0
    except ValueError as e:
        print(f"[explain] ERROR: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("=== SYSTEM ===")
        print(SYSTEM_PROMPT)
        print("\n=== USER ===")
        print(build_prompt(timeline, args.fps))
        return 0

    try:
        if args.offline:
            text, used_api = explain_offline(timeline, args.fps), False
        else:
            text, used_api = explain_with_api(timeline, args.fps,
                                              args.provider, args.model)
    except ValueError as e:
        print(f"[explain] ERROR: {e}", file=sys.stderr)
        return 1

    label = f"{args.provider} API" if used_api else "offline template"
    print(f"[explain] source: {label}\n")
    print(text)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n[explain] written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
