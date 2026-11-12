"""Session leases and two-player rooms for the native client."""
import json
import re
import secrets
import threading
import time
import unicodedata


LEASE_SECONDS = 60
ROOM_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CHARACTERS = {"orbit", "glider", "beat", "runner", "resident", "skater"}


class Lobby:
    def __init__(self):
        self.lock = threading.Lock()
        self.players = {}
        self.names = set()
        self.rooms = {}

    def leave_room(self, token):
        player = self.players[token]
        code = player.pop("room", "")
        if code:
            members = self.rooms[code]
            members.remove(token)
            if not members:
                del self.rooms[code]

    def disconnect(self, token):
        self.leave_room(token)
        self.names.remove(self.players.pop(token)["nickname"].casefold())

    def reserve_name(self, nickname):
        # Check every possible suffix at most once, in randomized order.
        start = secrets.randbelow(10000)
        step = secrets.randbelow(1000) * 10 + secrets.choice((1, 3, 7, 9))
        for index in range(10000):
            candidate = f"{nickname}{(start + index * step) % 10000:04d}"
            if candidate.casefold() not in self.names:
                return candidate
        return None

    def handle(self, action, data):
        token = data.get("token")
        if not isinstance(token, str) or not re.fullmatch("[0-9a-f]{32}", token):
            return 400, {"error": "잘못된 접속 정보입니다."}
        with self.lock:
            now = time.monotonic()
            for expired in [key for key, player in self.players.items() if player["expires"] <= now]:
                self.disconnect(expired)

            player = self.players.get(token)
            if action == "connect" and player is None:
                nickname = data.get("nickname")
                character = data.get("character")
                color = data.get("color", 0)
                if not isinstance(nickname, str) or not isinstance(character, str) or character not in CHARACTERS:
                    return 400, {"error": "주민과 닉네임을 확인해 주세요."}
                if type(color) is not int or not 0 <= color <= 10:
                    return 400, {"error": "선택한 색상을 확인해 주세요."}
                nickname = unicodedata.normalize("NFKC", nickname).strip()
                if not 1 <= len(nickname) <= 16 or not all(c.isalnum() or c in " _-" for c in nickname):
                    return 400, {"error": "닉네임은 1~16자의 글자, 숫자, 공백, 밑줄, 하이픈을 사용할 수 있어요."}
                if len(self.players) >= 1000:
                    return 503, {"error": "대기실이 가득 찼어요. 잠시 후 다시 시도해 주세요."}
                nickname = self.reserve_name(nickname)
                if nickname is None:
                    return 400, {"error": "다른 닉네임을 입력해 주세요."}
                player = self.players[token] = {"nickname": nickname, "character": character, "color": color}
                self.names.add(nickname.casefold())

            if action == "disconnect":
                if player is not None:
                    self.disconnect(token)
                return 200, {}
            if player is None:
                return 404, {"expired": True, "error": "접속이 종료됐어요. 다시 입장해 주세요."}

            player["expires"] = now + LEASE_SECONDS
            if action == "create" and "room" not in player:
                while True:
                    code = "".join(secrets.choice(ROOM_ALPHABET) for _ in range(6))
                    if code not in self.rooms:
                        break
                self.rooms[code] = [token]
                player["room"] = code
            elif action == "join":
                code = data.get("code")
                if not isinstance(code, str):
                    return 400, {"error": "참가 코드를 확인해 주세요."}
                code = code.strip().upper()
                if player.get("room") != code:
                    if code not in self.rooms:
                        return 404, {"error": "해당 코드의 방을 찾을 수 없어요."}
                    if len(self.rooms[code]) >= 2:
                        return 400, {"error": "방이 가득 찼어요."}
                    self.leave_room(token)
                    self.rooms[code].append(token)
                    player["room"] = code
            elif action == "leave":
                self.leave_room(token)

            code = player.get("room", "")
            members = [
                {field: self.players[key][field] for field in ("nickname", "character", "color")}
                for key in self.rooms.get(code, ())
            ]
            return 200, {"nickname": player["nickname"], "character": player["character"], "color": player["color"],
                         "code": code, "members": members,
                         "roomCount": len(self.rooms)}


lobby = Lobby()


def dispatch(path, length, stream):
    action = path.removeprefix("/lobby/")
    if action not in {"connect", "poll", "create", "join", "leave", "disconnect"}:
        return 404, {"error": "Not found"}
    try:
        size = int(length or "0")
        if not 0 < size <= 2048:
            return 413, {"error": "Invalid request size"}
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError()
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError()
    except (ValueError, TypeError, RecursionError):
        return 400, {"error": "Invalid lobby request"}
    return lobby.handle(action, data)
