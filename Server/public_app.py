"""Public WSGI gateway. Deploy one worker; credentials stay in server environment."""
import collections
from contextlib import closing
import datetime
import json
import os
import sqlite3
import threading
import time
import urllib.error
import server
import online_lobby
import online_songs

LIMIT_LOCK = threading.Lock()
RECENT = collections.deque()
DAILY_LIMIT = max(1, int(os.environ.get("DAILY_REQUEST_LIMIT", "400")))
MINUTE_LIMIT = max(1, int(os.environ.get("MINUTE_REQUEST_LIMIT", "20")))
QUOTA_DB = os.environ.get("QUOTA_DB", "/tmp/limit-breaker-quota.sqlite3")


def reserve_request():
    """Atomic shared daily budget across threads/restarts while the database survives."""
    now = time.monotonic()
    day = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    with LIMIT_LOCK:
        while RECENT and now - RECENT[0] >= 60:
            RECENT.popleft()
        if len(RECENT) >= MINUTE_LIMIT:
            return False
        with closing(sqlite3.connect(QUOTA_DB, timeout=3)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS quota(day TEXT PRIMARY KEY, used INTEGER NOT NULL)")
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO quota(day,used) VALUES (?,0)", (day,))
            used = db.execute("SELECT used FROM quota WHERE day=?", (day,)).fetchone()[0]
            if used >= DAILY_LIMIT:
                return False
            db.execute("UPDATE quota SET used=used+1 WHERE day=?", (day,))
            db.execute("DELETE FROM quota WHERE day<?", (day,))
            db.commit()
        RECENT.append(now)
        return True


def application(environ, start_response):
    def reply(code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        statuses = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
                    413: "Payload Too Large", 415: "Unsupported Media Type", 429: "Too Many Requests", 503: "Service Unavailable"}
        headers = [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body))),
                   ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff")]
        if code == 429:
            headers.append(("Retry-After", "60"))
        start_response(str(code) + " " + statuses[code], headers)
        return [body]
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "")
    if path == "/songs" or path.startswith("/songs/"):
        return online_songs.application(environ, start_response, reply, reserve_request)
    if path.startswith("/lobby/"):
        if method != "POST":
            return reply(405, {"error": "POST required"})
        if environ.get("CONTENT_TYPE", "").split(";")[0].lower() != "application/json":
            return reply(415, {"error": "JSON required"})
        return reply(*online_lobby.dispatch(path, environ.get("CONTENT_LENGTH"), environ["wsgi.input"]))
    if path == "/health" and method == "GET":
        return reply(200 if os.environ.get("GEMINI_API_KEY") else 503,
                     {"ready": bool(os.environ.get("GEMINI_API_KEY")), "protocol": server.PROTOCOL, "diagnosticsVersion": 1})
    if path != "/analyze":
        return reply(404, {"error": "Not found"})
    if method != "POST":
        return reply(405, {"error": "POST required"})
    if environ.get("CONTENT_TYPE", "").split(";")[0].lower() != "application/json":
        return reply(415, {"error": "JSON required"})
    try:
        length = int(environ.get("CONTENT_LENGTH", "0"))
        if not 0 < length <= 131072:
            return reply(413, {"error": "Invalid request size"})
        raw = environ["wsgi.input"].read(length)
        if len(raw) != length:
            raise ValueError()
        data = json.loads(raw.decode("utf-8"))
        server.validate_stats(data)
    except (ValueError, TypeError, AttributeError, KeyError, UnicodeError):
        return reply(400, {"error": "Invalid gameplay summary"})
    try:
        if not reserve_request():
            return reply(429, {"error": "Request limit reached"})
        return reply(200, server.analyze(data))
    except server.RequestThrottled:
        return reply(429, {"error": "Analysis busy"})
    except urllib.error.HTTPError as error:
        return reply(503, server.diagnostic_failure(error))
    except Exception as error:
        # Never return exception strings, upstream bodies, headers or secrets.
        return reply(503, server.diagnostic_failure(error))
