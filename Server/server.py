"""Loopback-only Gemini bridge. No third-party packages or key files."""
import json
import os
import threading
import math
import logging
import time
from concurrent.futures import Future
from contextlib import contextmanager, closing
from datetime import datetime, timedelta, timezone
import sqlite3
import tempfile
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SKILLS = ("Tap", "Chord", "Hold", "Switch", "Repeat")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
LOCK = threading.Lock()
CACHE = {}
CALLS = 0
INFLIGHT = {}
LAST_REQUEST_AT = float("-inf")
MIN_REQUEST_INTERVAL = 3.0
PROTOCOL = 6
KST = timezone(timedelta(hours=9))
AI_USAGE_DB = os.environ.get("AI_USAGE_DB", os.path.join(tempfile.gettempdir(), "beatlimit-ai-usage.sqlite3"))


def usage_connection():
    db = sqlite3.connect(AI_USAGE_DB, timeout=2)
    try:
        db.execute("""CREATE TABLE IF NOT EXISTS ai_usage(
            id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, kind TEXT NOT NULL, result TEXT NOT NULL)""")
        db.execute("CREATE INDEX IF NOT EXISTS ai_usage_time ON ai_usage(timestamp)")
        return db
    except Exception:
        db.close()
        raise


@contextmanager
def record_ai_call(kind):
    started = datetime.now(KST)
    result = "failed"
    try:
        yield
        result = "success"
    finally:
        try:
            with closing(usage_connection()) as db, db:
                db.execute("INSERT INTO ai_usage(timestamp,kind,result) VALUES (?,?,?)",
                           (started.isoformat(timespec="milliseconds"), kind, result))
                db.execute("DELETE FROM ai_usage WHERE timestamp < ?",
                           ((started - timedelta(days=30)).isoformat(timespec="milliseconds"),))
        except sqlite3.Error:
            logging.warning("AI_USAGE_WRITE_FAILED")


def ai_usage():
    today = datetime.now(KST).date().isoformat()
    with closing(usage_connection()) as db:
        counts = dict(db.execute("SELECT kind,COUNT(*) FROM ai_usage WHERE timestamp >= ? GROUP BY kind", (today,)))
        rows = db.execute("SELECT timestamp,kind,result FROM ai_usage ORDER BY timestamp DESC,id DESC LIMIT 100")
        return {"date": today, "playCalls": counts.get("play", 0), "songCalls": counts.get("song", 0),
                "entries": [dict(timestamp=stamp, kind=kind, result=result) for stamp, kind, result in rows]}

TARGETS = {1: (90, 95), 2: (90, 95), 3: (82, 90), 4: (75, 82), 5: (68, 75), 6: (58, 61)}


def number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def validate_rows(stats):
    if not isinstance(stats, list) or len(stats) != 5:
        raise ValueError("Expected five skill summaries")
    seen = set()
    for row in stats:
        if row.get("skill") not in SKILLS or row["skill"] in seen:
            raise ValueError("Invalid skill")
        seen.add(row["skill"])
        for name in ("count", "successes"):
            if type(row.get(name)) is not int or not 0 <= row[name] <= 100000:
                raise ValueError("Invalid count")
        if row["successes"] > row["count"] or not number(row.get("meanErrorMs"), 0, 100000):
            raise ValueError("Invalid performance")
    return [{k: row[k] for k in ("skill", "count", "successes", "meanErrorMs")} for row in stats]


def validate_input(data):
    if data is None:
        return None  # Legacy summaries have no key-level evidence.
    if not isinstance(data, dict): raise ValueError("Invalid input analysis")
    result = {}
    for field, names in (("keys", ("D", "F", "J", "K")), ("groups", ("DF", "JK"))):
        rows = data.get(field)
        if not isinstance(rows, list) or len(rows) != len(names): raise ValueError("Invalid key rows")
        cleaned = []
        for row, name in zip(rows, names):
            if not isinstance(row, dict) or row.get("key") != name: raise ValueError("Invalid key order")
            counts = ("count", "successes", "presses", "wrongPresses", "earlyPresses", "emptyPresses")
            if any(type(row.get(k)) is not int or not 0 <= row[k] <= 1000000 for k in counts):
                raise ValueError("Invalid key counts")
            if row["successes"] > row["count"] or sum(row[k] for k in counts[3:]) > row["presses"]:
                raise ValueError("Inconsistent key counts")
            if not number(row.get("meanErrorMs"), 0, 100000): raise ValueError("Invalid key timing")
            cleaned.append({k: row[k] for k in ("key", *counts, "meanErrorMs")})
        result[field] = cleaned
    for group, keys in zip(result["groups"], (result["keys"][:2], result["keys"][2:])):
        for k in ("count", "successes", "presses", "wrongPresses", "earlyPresses", "emptyPresses"):
            if group[k] != sum(row[k] for row in keys): raise ValueError("Inconsistent group totals")
    matrix = data.get("confusion")
    ambiguous = data.get("ambiguousWrongPresses")
    if not isinstance(matrix, list) or len(matrix) != 16 or any(type(n) is not int or not 0 <= n <= 1000000 for n in matrix):
        raise ValueError("Invalid confusion matrix")
    if type(ambiguous) is not int or not 0 <= ambiguous <= 1000000:
        raise ValueError("Invalid ambiguous count")
    if any(matrix[i*4+i] for i in range(4)) or sum(matrix)+ambiguous != sum(r["wrongPresses"] for r in result["keys"]):
        raise ValueError("Inconsistent confusion matrix")
    presses = data.get("recentPresses", [])
    if not isinstance(presses, list) or len(presses) > 32: raise ValueError("Too many recent presses")
    cleaned = []
    for press in presses:
        if not isinstance(press, dict): raise ValueError("Invalid press")
        for k, low, high in (("stage",1,6), ("pressedKey",0,3), ("expectedMask",0,15)):
            if type(press.get(k)) is not int or not low <= press[k] <= high: raise ValueError("Invalid press key")
        if not number(press.get("timeSeconds"),0,600) or not number(press.get("errorMs"),-100000,100000):
            raise ValueError("Invalid press time")
        if press.get("outcome") not in ("Hit", "HoldStart", "WrongKey", "Empty", "TooEarly", "Late"):
            raise ValueError("Invalid press outcome")
        cleaned.append({k: press[k] for k in ("stage","pressedKey","expectedMask","timeSeconds","errorMs","outcome")})
    return {**result, "confusion": matrix, "ambiguousWrongPresses": ambiguous, "recentPresses": cleaned}


def validate_judgements(value, count):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 5 or any(type(n) is not int or not 0 <= n <= 1000000 for n in value):
        raise ValueError("Invalid judgement distribution")
    if sum(value) != count:
        raise ValueError("Inconsistent judgement total")
    return value


def validate_stats(data):
    if not isinstance(data, dict) or data.get("phase") not in ("training", "final"):
        raise ValueError("Invalid phase")
    stage = data.get("nextStage")
    if type(stage) is not int or stage not in TARGETS:
        raise ValueError("Invalid next stage")
    if not number(data.get("durationSeconds"), 5, 600) or not number(data.get("bpm"), 30, 300):
        raise ValueError("Invalid song context")
    rounds = data.get("rounds", [])
    if not isinstance(rounds, list) or len(rounds) > 20:
        raise ValueError("Invalid round count")
    cleaned_rounds = []
    for row in rounds:
        if type(row.get("stage")) is not int or not 1 <= row["stage"] <= 6:
            raise ValueError("Invalid previous stage")
        for name, low, high in (("count", 0, 100000), ("successes", 0, 100000), ("perfects", 0, 100000),
                ("finalEnergy", 0, 100), ("noteSpeed", 0, 22), ("groupGapBeats", 0, 2),
                ("meanErrorMs", 0, 100000), ("burstCount", 0, 4), ("chordSize", 0, 4)):
            if not number(row.get(name), low, high): raise ValueError("Invalid previous settings")
        if not row["perfects"] <= row["successes"] <= row["count"]:
            raise ValueError("Invalid previous counts")
        clean = {k: row[k] for k in ("stage", "count", "successes", "perfects", "finalEnergy", "noteSpeed",
            "groupGapBeats", "meanErrorMs", "burstCount", "chordSize")}
        clean["stats"] = validate_rows(row.get("stats"))
        clean["input"] = validate_input(row.get("input"))
        clean["judgements"] = validate_judgements(row.get("judgements"), row["count"])
        if clean["judgements"] is not None:
            grades = clean["judgements"]
            if sum(grades[:3]) != row["successes"] or grades[0] != row["perfects"]:
                raise ValueError("Inconsistent judgement successes")
        variety = row.get("rhythmVariety", 0)
        if type(variety) is not int or not 0 <= variety <= 3: raise ValueError("Invalid previous rhythm variety")
        clean["rhythmVariety"] = variety
        cleaned_rounds.append(clean)
    return {"phase": data["phase"], "nextStage": stage, "durationSeconds": data["durationSeconds"],
        "bpm": data["bpm"], "stats": validate_rows(data.get("stats")), "rounds": cleaned_rounds,
        "input": validate_input(data.get("input")),
        "judgements": validate_judgements(data.get("judgements"), sum(row["count"] for row in data["stats"]))}


def guide_text(text, field):
    if not isinstance(text, str):
        raise ValueError("Invalid guide message")
    clean = " ".join(text.split())
    if not 1 <= len(clean) <= 100 or any(c in clean for c in "<>%") or any(c.isdigit() for c in clean):
        logging.warning("AI_ROUTE_TEXT_NORMALIZED field=%s", field)
        return "플레이 기록을 참고해 다음 경로를 준비했어요." if field == "message" else "박자와 입력 기록에 맞춰 경로를 조정했어요."
    return clean


def validate_plan(plan, stats, next_stage=2):
    eligible = {r["skill"] for r in stats if r["count"] >= 3 and r["skill"] in ("Tap", "Hold")}
    focus = plan.get("focusSkills")
    if not isinstance(focus, list) or not 1 <= len(focus) <= 2 or len(set(focus)) != len(focus):
        raise ValueError("Invalid focus list")
    if any(s not in eligible for s in focus):
        raise ValueError("Unobserved focus skill")
    dialogue = {field: guide_text(plan.get(field), field) for field in ("message", "adjustment")}
    if plan.get("stage") != next_stage or type(plan.get("stage")) is not int:
        raise ValueError("Wrong next stage")
    for field, low, high in (("noteSpeed",6,22), ("burstCount",2,4), ("chordSize",2,4)):
        if not number(plan.get(field),low,high): raise ValueError("Invalid chart settings")
    if type(plan["burstCount"]) is not int or type(plan["chordSize"]) is not int or type(plan.get("dualHolds")) is not bool:
        raise ValueError("Invalid setting type")
    for field, allowed in (("groupGapBeats",(0.5,0.75,1,1.5,2)), ("burstIntervalBeats",(0.25,0.5)),
            ("holdBeats",(0.5,0.75,1,1.5))):
        if not number(plan.get(field),0.01,2) or plan[field] not in allowed: raise ValueError("Invalid beat grid")
    fields = ("focusSkills", "message", "adjustment", "stage", "noteSpeed", "groupGapBeats", "burstCount",
        "burstIntervalBeats", "chordSize", "holdBeats", "dualHolds")
    weights = plan.get("laneWeights", [1, 1, 1, 1])
    if not isinstance(weights, list) or len(weights) != 4 or any(type(w) is not int or not 1 <= w <= 3 for w in weights):
        raise ValueError("Invalid lane weights")
    variety = plan.get("rhythmVariety", 1)
    if type(variety) is not int or not 1 <= variety <= 3: raise ValueError("Invalid rhythm variety")
    return {**{field: plan[field] for field in fields}, **dialogue, "laneWeights": weights, "rhythmVariety": variety, "source": "gemini"}


class RequestThrottled(Exception):
    def __init__(self, seconds):
        self.retry_after = max(1, math.ceil(seconds))


def failure_info(error):
    """Expose only retry metadata, never upstream response text or credentials."""
    info = {"error": "AI unavailable", "upstreamStatus": error.code}
    try:
        payload = json.loads(error.read(65536)).get("error", {})
        details = payload.get("details", [])
        for detail in details:
            for violation in detail.get("violations", []):
                metric = (violation.get("quotaId", "") + " " + violation.get("quotaMetric", "")).lower()
                if "perday" in metric or "per_day" in metric or "daily" in metric:
                    info["quotaExhausted"] = True
            delay = detail.get("retryDelay", "")
            if isinstance(delay, str) and delay.endswith("s"):
                info["retryAfterSeconds"] = min(86400, max(0, float(delay[:-1])))
    except (ValueError, TypeError, AttributeError):
        pass
    return info


def diagnostic_failure(error):
    """Stable diagnostic categories; exception text and model dialogue remain private."""
    if isinstance(error, urllib.error.HTTPError):
        info = failure_info(error)
        code = "UPSTREAM_HTTP"
    else:
        info = {"error": "AI unavailable"}
        if isinstance(error, TimeoutError): code = "UPSTREAM_TIMEOUT"
        elif isinstance(error, urllib.error.URLError): code = "UPSTREAM_NETWORK"
        elif isinstance(error, json.JSONDecodeError): code = "RESPONSE_JSON"
        elif isinstance(error, ValueError):
            code = {"Invalid guide message": "RESPONSE_TEXT", "Incomplete response": "RESPONSE_INCOMPLETE"}.get(str(error), "RESPONSE_SCHEMA")
        elif isinstance(error, (KeyError, IndexError, TypeError)): code = "RESPONSE_SHAPE"
        elif isinstance(error, RuntimeError) and str(error) == "GEMINI_API_KEY missing": code = "SERVER_CONFIGURATION"
        else: code = "SERVER_INTERNAL"
    info["errorCode"] = code
    logging.warning("AI_ROUTE_FAILURE code=%s upstream=%s", code, info.get("upstreamStatus", "not-reported"))
    return info


def analyze(data):
    global CALLS, LAST_REQUEST_AT
    clean = validate_stats(data)
    cache_key = json.dumps(clean, sort_keys=True)
    with LOCK:
        if cache_key in CACHE:
            return CACHE[cache_key]
        pending = INFLIGHT.get(cache_key)
        owner = pending is None
        if owner:
            # Different requests cannot run upstream at the same time.
            if INFLIGHT:
                raise RequestThrottled(MIN_REQUEST_INTERVAL)
            remaining = MIN_REQUEST_INTERVAL - (time.monotonic() - LAST_REQUEST_AT)
            if remaining > 0:
                raise RequestThrottled(remaining)
            key = os.environ.get("GEMINI_API_KEY", "")
            if not key:
                raise RuntimeError("GEMINI_API_KEY missing")
            pending = Future()
            INFLIGHT[cache_key] = pending
            LAST_REQUEST_AT = time.monotonic()
            CALLS += 1  # Monitoring only; no total-count ceiling.
    if not owner:
        return pending.result(timeout=23)
    try:
        plan = fetch_plan(clean, key)
        with LOCK:
            if len(CACHE) >= 256:
                CACHE.pop(next(iter(CACHE)))
            CACHE[cache_key] = plan
        pending.set_result(plan)
        return plan
    except Exception as error:
        pending.set_exception(error)
        raise
    finally:
        with LOCK:
            INFLIGHT.pop(cache_key, None)


def fetch_plan(clean, key):
    cache_key = json.dumps(clean, sort_keys=True)
    eligible = [row["skill"] for row in clean["stats"] if row["count"] >= 3 and row["skill"] in ("Tap", "Hold")]
    if not eligible:
        raise ValueError("Insufficient observations")
    properties = {
        "focusSkills": {"type": "ARRAY", "items": {"type": "STRING", "enum": eligible}, "minItems": 1, "maxItems": min(2,len(eligible))},
        "message": {"type": "STRING"}, "adjustment": {"type": "STRING"}, "stage": {"type": "INTEGER", "minimum": clean["nextStage"], "maximum": clean["nextStage"]},
        "noteSpeed": {"type": "NUMBER", "minimum": 6, "maximum": 22}, "groupGapBeats": {"type": "NUMBER", "minimum": .5, "maximum": 2},
        "burstCount": {"type": "INTEGER", "minimum": 2, "maximum": 4}, "burstIntervalBeats": {"type": "NUMBER", "minimum": .25, "maximum": .5},
        "chordSize": {"type": "INTEGER", "minimum": 2, "maximum": 4}, "holdBeats": {"type": "NUMBER", "minimum": .5, "maximum": 1.5}, "dualHolds": {"type": "BOOLEAN"},
        "laneWeights": {"type": "ARRAY", "items": {"type": "INTEGER", "minimum": 1, "maximum": 3}, "minItems": 4, "maxItems": 4},
        "rhythmVariety": {"type": "INTEGER", "minimum": 1, "maximum": 3}}
    schema = {"type": "OBJECT", "properties": properties, "required": list(properties)}
    low, high = TARGETS[clean["nextStage"]]
    prompt = ("You are the sole difficulty designer of a four-key rhythm game. Review ALL skill summaries and per-round records. "
              "You must choose actual next-stage chart parameters based on observed ability, previous settings, energy and timing errors. "
              f"INTERNAL ONLY: aim for final escape ENERGY {low} to {high}, not note accuracy. "
              "Energy starts at 30. Judgements are PERFECT/GREAT/GOOD/BAD/MISS; their energy changes are +5/+3/+1/-3/-6, clamped 0..100. Clear threshold is always 60. "
              "All stages use absolute timing windows PERFECT<=46ms, GREAT<=83ms, GOOD<=118ms, BAD<=148ms, otherwise MISS. These are rounded means of normalized osu!mania OD5, StepMania default and Quaver Standard windows. The visible judgement band highlights PERFECT and GREAT only, up to 83ms; GOOD and BAD remain valid outside the highlight. Do not tighten timing windows for the final stage; adjust chart parameters instead. "
              "judgements arrays use [PERFECT,GREAT,GOOD,BAD,MISS] counts. BAD and MISS break combo and are not successes; only the first three are successes. "
              "Scores are +100/+90/+70/+30/-30 for those grades, floored at zero. Use the full grade distribution when available. "
              "Use the observed grade shares plus upcoming duration/BPM to estimate energy. Never alter the gauge or score. "
              "If previous energy is higher than next target, increase challenge; if clearly lower, ease it. Use all rounds and all skills, with recent performance weighted more. "
              "Choose 1 or 2 observed skills to introduce first; all generated notes are Tap or Hold; simultaneous and repeated taps are rhythmic patterns, not separate note types. "
              "Never recommend a weak-skills-only challenge. "
              "Analyze input.keys and input.groups (DF and JK), including each round's input, with recent rounds weighted more. "
              "count/successes measure opportunities and completed hits, presses measure actual key-downs. Compare success rates per opportunity, not raw press totals. "
              "WrongKey means another lane was hittable but the pressed lane was not; this is evidence of possible key confusion, not proof of intent. "
              "confusion is a row-major 4x4 matrix [expected D,F,J,K][pressed D,F,J,K]; ambiguous chord mistakes are counted separately. "
              "recentPresses records stage/timeSeconds/pressedKey (D=0,F=1,J=2,K=3), expectedMask (bits D=1,F=2,J=4,K=8), signed errorMs and outcome. "
              "Empty inputs can be extra taps; TooEarly indicates timing rather than wrong-key choice. HoldStart is not a completed hold success. "
              "Never infer actual physical hand use from these key groups. Missing input data or fewer than three opportunities per key is insufficient evidence. "
              "If one group misses more or key confusion is frequent, ease speed, spacing, bursts or chords first and provide gentle balanced practice. "
              "Choose laneWeights as four integers 1..3 for D,F,J,K; equal weights mean balanced distribution. "
              "With sufficient evidence, moderately increase weights of weak keys for practice while easing other parameters; never remove strong keys or overwhelm weak keys. "
              "Weights affect Tap and Hold lane choices; simultaneous taps use shuffled lane pairs. "
              "Choose rhythmVariety integer 1..3 from observed ability, independently of speed. "
              "Level 1 shuffles skill order, lane choices, chord pairs and switch directions while keeping readable spacing. "
              "Level 2 also mixes burst sizes, hold lengths, offbeat tap/chord follow-ups and quarter-beat rests. "
              "Level 3 adds quarter-beat pickups, phrase density changes and free-lane taps during holds. "
              "When performance is strong, raise rhythmic complexity or density rather than only raising speed. "
              "When confusion or misses rise, simplify rhythmVariety and chords/bursts while preserving varied key practice. "
              "Base both the difficulty and your explanation on observed timing, key confusion and group success rates; do not invent observations. "
              "Bounds: noteSpeed 6..22 world units/sec; groupGapBeats one of [0.5,0.75,1,1.5,2]; "
              "burstCount integer 2..4; burstIntervalBeats [0.25,0.5]; chordSize integer 2..4; holdBeats [0.5,0.75,1,1.5]; dualHolds boolean. "
              "Faster speed reduces preview time only (travel distance 9); shorter gaps, bigger chords/bursts increase physical difficulty. "
              "Chart starts at beat 4 and ends two seconds before audio end. Skill order changes in shuffled five-skill bags, not a fixed cycle. "
              "A block reserves at least groupGapBeats and a quarter beat after its last note or hold end; same-lane overlaps are forbidden. "
              "Also roleplay the friendly digital guide Lumi helping the player escape a neon digital world. "
              "Provide message as Lumi's player-facing Korean dialogue in gentle polite speech, under 70 characters and at most two short sentences. "
              "Base it on observed gameplay AND the actual chosen next settings versus previous settings. "
              "Naturally preview whether the next path may feel a little harder, challenging, or more manageable, then encourage the player. "
              "Do not always claim it gets harder when you actually eased the settings. "
              "Only nextStage 6 is the final path. For nextStage 2 through 5, describe the upcoming route and avoid wording that implies the journey is ending, including 마지막 and 최종. "
              "For nextStage 6, encourage using everything learned. "
              "Speak directly to the player without speaker labels, model names, technical parameter names, or diagnostic jargon. "
              "Examples of tone only, never fixed output: 다음 경로는 조금 더 어려울 수도 있어요! 네 리듬을 믿어 봐요. "
              "Provide adjustment as an internal Korean reason matching chosen settings, under 70 chars. "
              "Do not reveal target energy, probabilities, numbers, percentages, or difficulty engineering intent in either Korean text. "
              "Do not invent evidence or generate note timestamps. Summarized gameplay data:\n" + cache_key)
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {
        "responseMimeType": "application/json", "responseSchema": schema,
        "maxOutputTokens": 1000, "temperature": 0.2}}
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/" + MODEL + ":generateContent",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with record_ai_call("play"):
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.load(response)
        candidate = result["candidates"][0]
        if candidate.get("finishReason") != "STOP":
            raise ValueError("Incomplete response")
        raw = "".join(p.get("text", "") for p in candidate["content"]["parts"] if not p.get("thought"))
        return validate_plan(json.loads(raw), clean["stats"], clean["nextStage"])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # Never log bodies, headers or API keys.

    def reply(self, status, data):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/songs" or self.path.startswith("/songs/"):
            self.serve_online_songs("GET")
        elif self.path == "/health":
            self.reply(200, {"ready": bool(os.environ.get("GEMINI_API_KEY")), "model": MODEL, "protocol": PROTOCOL,
                "requestsSent": CALLS, "minRequestIntervalSeconds": MIN_REQUEST_INTERVAL})
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/songs":
            self.serve_online_songs("POST")
            return
        if self.path.startswith("/lobby/"):
            if self.headers.get("Origin") or self.headers.get_content_type() != "application/json":
                self.reply(403, {"error": "Native local client only"}); return
            import online_lobby
            self.reply(*online_lobby.dispatch(self.path, self.headers.get("Content-Length"), self.rfile))
            return
        if self.path != "/analyze":
            self.reply(404, {"error": "Not found"}); return
        if self.headers.get("Origin") or self.headers.get_content_type() != "application/json":
            self.reply(403, {"error": "Native local client only"}); return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 131072:
                raise ValueError("Invalid payload size")
            data = json.loads(self.rfile.read(size))
            validate_stats(data)
        except (ValueError, TypeError, AttributeError):
            self.reply(400, {"error": "Invalid gameplay summary"}); return
        try:
            self.reply(200, analyze(data))
        except RequestThrottled as error:
            self.reply(429, {"error": "Local request spacing", "retryAfterSeconds": error.retry_after})
        except urllib.error.HTTPError as error:
            self.reply(503, diagnostic_failure(error))
        except Exception as error:
            self.reply(503, diagnostic_failure(error))

    def serve_online_songs(self, method):
        import online_songs

        if self.headers.get("Origin"):
            self.reply(403, {"error": "Native local client only"})
            return

        def reply(status, payload):
            self.reply(status, payload)
            return ()

        def start_response(status, headers):
            self.send_response(int(status.split()[0]))
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()

        environ = {
            "PATH_INFO": self.path, "REQUEST_METHOD": method, "wsgi.input": self.rfile,
            "CONTENT_LENGTH": self.headers.get("Content-Length"),
            "HTTP_AUTHORIZATION": self.headers.get("Authorization", ""),
            "HTTP_X_SONG_FORMAT": self.headers.get("X-Song-Format", ""),
            "HTTP_X_SONG_TITLE": self.headers.get("X-Song-Title", ""),
        }
        for block in online_songs.application(environ, start_response, reply, lambda: True):
            self.wfile.write(block)


if __name__ == "__main__":
    print("LIMIT BREAKER AI bridge: http://127.0.0.1:8765 (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
