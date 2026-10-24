"""Minimal client for IG's REST API, restricted to IG's DEMO environment.

Only the demo gateway is ever contacted: paper trading with demo money. There is no setting
that points it at a live account.

Credentials come from the environment (.env): IG_API_KEY, IG_USERNAME, IG_PASSWORD and
optionally IG_ACCOUNT_ID.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

DEMO_URL = "https://demo-api.ig.com/gateway/deal"
RESOLUTIONS = {"MINUTE_15", "MINUTE_30", "HOUR", "HOUR_2", "HOUR_4", "DAY"}


class IGError(RuntimeError):
    pass


class IGNotConfigured(IGError):
    pass


class IGAllowanceExceeded(IGError):
    """IG limits how much price history an account may download per week."""


class IGLoginError(IGError):
    """Wrong login details (or login refused). Not retried: repeated failures get the API key
    rate-limited and can lock the account."""


class IGRateLimited(IGError):
    """Too many requests for the API key / account in the last minute."""


# After a failed login, don't try again for a while (per username, across clients in this process).
LOGIN_BACKOFF = 15 * 60
# IG sessions expire after 6 hours idle; log in again well before that
SESSION_MAX_AGE = 4 * 3600
_login_blocked: dict[str, tuple[float, str]] = {}
# Seconds between requests, across every process on this machine (web server, jobs, trading
# cycles): IG's demo API allows roughly 30 non-trading requests a minute per API key.
MIN_INTERVAL = 2.1
THROTTLE_FILE = Path(tempfile.gettempdir()) / "energyprediction-ig-throttle"


def _login_hint(code: str) -> str:
    c = code.lower()
    if "identifier" in c:
        return (" — IG_USERNAME must be your IG *username* (shown in My IG and on the demo account), "
                "not your email address")
    if "invalid-details" in c or "invalid.details" in c or "credentials" in c:
        return (" — IG doesn't recognise this username/password on the DEMO platform. Use the demo account's "
                "own login (IG demo accounts have a separate username and password from a live account); "
                "check you can log in to the demo web platform with exactly these")
    if "api-key" in c and ("invalid" in c or "disabled" in c):
        return " — IG_API_KEY not recognised: create it while logged in to the DEMO account and make sure it's enabled"
    if "allowance" in c:
        return " — too many requests for this API key; wait a few minutes before trying again"
    return ""


def ig_configured() -> bool:
    return all(os.environ.get(k) for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD"))


class IGClient:
    """Session handling, market lookup, price history, quotes and (demo) orders."""

    def __init__(self, api_key: str | None = None, username: str | None = None, password: str | None = None,
                 account_id: str | None = None, session: requests.Session | None = None, timeout: float = 20.0,
                 session_file: str | Path | None = None):
        self.api_key = api_key or os.environ.get("IG_API_KEY")
        self.username = username or os.environ.get("IG_USERNAME")
        self.password = password or os.environ.get("IG_PASSWORD")
        self.account_id = account_id or os.environ.get("IG_ACCOUNT_ID") or None
        if not (self.api_key and self.username and self.password):
            raise IGNotConfigured("IG demo login not set: add IG_API_KEY, IG_USERNAME and IG_PASSWORD to .env")
        self.http = session or requests.Session()
        self.timeout = timeout
        self.base = DEMO_URL
        self._tokens: dict[str, str] = {}
        self._login_at = 0.0
        self._lock = threading.Lock()
        self.account: dict = {}
        self.last_allowance: dict | None = None
        # Optional file to share the login between processes (each trading cycle is a new process;
        # without it a 1-minute interval would mean ~1,400 logins a day and IG would lock the key).
        self.session_file = Path(session_file) if session_file else None
        self._load_session()

    # ── low level ────────────────────────────────────────────────────────
    _last_request = 0.0  # fallback when the shared throttle file can't be used

    def _throttle(self) -> None:
        """Space requests MIN_INTERVAL apart across processes (a small lock file holds the time of
        the last request), so a data refresh, a backtest and a trading cycle running together
        still stay under IG's per-minute limit."""
        if MIN_INTERVAL <= 0:
            return
        try:
            import fcntl

            with open(THROTTLE_FILE, "a+") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                fh.seek(0)
                try:
                    last = float(fh.read().strip() or 0)
                except ValueError:
                    last = 0.0
                wait = last + MIN_INTERVAL - time.time()
                if wait > 0:
                    time.sleep(wait)
                fh.seek(0)
                fh.truncate()
                fh.write(str(time.time()))
                fh.flush()
                fcntl.flock(fh, fcntl.LOCK_UN)
            return
        except (ImportError, OSError):
            pass
        wait = IGClient._last_request + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        IGClient._last_request = time.time()

    def _headers(self, version: int, tokens: bool = True) -> dict:
        h = {"X-IG-API-KEY": self.api_key, "Content-Type": "application/json; charset=UTF-8",
             "Accept": "application/json; charset=UTF-8", "Version": str(version)}
        if tokens:
            h.update(self._tokens)
        return h

    # ── shared session ───────────────────────────────────────────────────
    def _session_owner(self) -> str:
        return hashlib.sha256(f"{self.api_key}|{self.username}|{self.account_id or ''}".encode()).hexdigest()[:16]

    def _load_session(self) -> None:
        if not self.session_file:
            return
        try:
            data = json.loads(self.session_file.read_text())
        except (OSError, ValueError):
            return
        age = time.time() - float(data.get("login_at", 0))
        if data.get("owner") == self._session_owner() and 0 <= age < SESSION_MAX_AGE and data.get("tokens"):
            self._tokens, self._login_at = dict(data["tokens"]), float(data["login_at"])
            self.account = data.get("account") or {}

    def _save_session(self) -> None:
        if not self.session_file:
            return
        try:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.session_file.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # tokens: owner-only
            with os.fdopen(fd, "w") as fh:
                json.dump({"owner": self._session_owner(), "login_at": self._login_at,
                           "tokens": self._tokens, "account": self.account}, fh)
            os.replace(tmp, self.session_file)
        except OSError as exc:
            log.debug("could not save IG session: %s", exc)

    def login(self) -> dict:
        with self._lock:
            if "@" in self.username:  # IG rejects emails here; don't spend a login attempt on it
                raise IGLoginError("IG_USERNAME looks like an email address" + _login_hint("identifier"))
            blocked = _login_blocked.get(self.username)
            if blocked and time.time() < blocked[0]:
                mins = (blocked[0] - time.time()) / 60
                raise IGLoginError(f"IG demo login failed recently ({blocked[1]}); not retrying for another "
                                   f"{mins:.0f} min (fix .env, then restart the server to retry sooner)")
            self._throttle()
            resp = self.http.post(f"{self.base}/session", json={"identifier": self.username, "password": self.password},
                                  headers=self._headers(2, tokens=False), timeout=self.timeout)
            if resp.status_code != 200:
                code = _error_code(resp)
                msg = f"{code}{_login_hint(code)}"
                _login_blocked[self.username] = (time.time() + LOGIN_BACKOFF, msg)
                raise IGLoginError(f"IG demo login failed ({resp.status_code}): {msg}")
            _login_blocked.pop(self.username, None)
            self._tokens = {"CST": resp.headers.get("CST", ""), "X-SECURITY-TOKEN": resp.headers.get("X-SECURITY-TOKEN", "")}
            self._login_at = time.time()
            body = resp.json()
            self.account = {"account_id": body.get("currentAccountId"), "currency": body.get("currencyIsoCode"),
                            "accounts": [{"id": a.get("accountId"), "name": a.get("accountName"),
                                          "type": a.get("accountType")} for a in body.get("accounts", [])]}
        if self.account_id and self.account_id != self.account.get("account_id"):
            self.request("PUT", "/session", 1, {"accountId": self.account_id, "defaultAccount": False})
            self.account["account_id"] = self.account_id
        self._save_session()
        log.info("IG demo: logged in (account %s)", self.account.get("account_id"))
        return self.account

    def request(self, method: str, path: str, version: int = 1, body: Any = None, params: dict | None = None,
                extra_headers: dict | None = None) -> Any:
        # sessions last 6 hours without activity; refresh well before that
        if not self._tokens or time.time() - self._login_at > SESSION_MAX_AGE:
            self.login()
        for attempt in range(3):
            headers = self._headers(version) | (extra_headers or {})
            self._throttle()
            resp = self.http.request(method, f"{self.base}{path}", json=body, params=params, headers=headers,
                                     timeout=self.timeout)
            if resp.status_code == 401 and attempt == 0:  # token expired
                self.login()
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            break
        if resp.status_code >= 400:
            code = _error_code(resp)
            if "historical-data-allowance" in code.lower():
                raise IGAllowanceExceeded(f"IG price-history allowance used up ({code}); it resets weekly")
            if "allowance" in code.lower():
                raise IGRateLimited(f"IG request limit reached ({code}); try again in a minute")
            raise IGError(f"IG {method} {path} failed ({resp.status_code}): {code}")
        return resp.json() if resp.content else {}

    # ── markets ──────────────────────────────────────────────────────────
    def search_markets(self, term: str) -> list[dict]:
        return self.request("GET", "/markets", 1, params={"searchTerm": term}).get("markets", [])

    def market(self, epic: str) -> dict:
        return self.request("GET", f"/markets/{epic}", 3)

    def find_market(self, term: str, prefer: tuple[str, ...] = ("UK", "NBP")) -> dict:
        """Pick the best match for e.g. 'UK Natural Gas': an undated (rolling cash) market whose
        name mentions UK/NBP, otherwise the first commodity match."""
        found = [m for m in self.search_markets(term) if m.get("epic")]
        if not found:
            raise IGError(f"no IG market found for {term!r}")

        def score(m):
            name = (m.get("instrumentName") or "").upper()
            return (
                -sum(p in name for p in prefer),
                0 if m.get("expiry") in ("-", "DFB", None) else 1,  # undated first
                0 if (m.get("instrumentType") or "").upper() == "COMMODITIES" else 1,
                len(name),
            )

        return sorted(found, key=score)[0]

    def quote(self, epic: str) -> dict:
        m = self.market(epic)
        snap, inst = m.get("snapshot", {}), m.get("instrument", {})
        return {"epic": epic, "bid": _f(snap.get("bid")), "offer": _f(snap.get("offer")),
                "status": snap.get("marketStatus"), "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "name": inst.get("name"), "expiry": inst.get("expiry")}

    def instrument(self, epic: str) -> dict:
        m = self.market(epic)
        inst, rules = m.get("instrument", {}), m.get("dealingRules", {})
        currency = next((c.get("code") for c in inst.get("currencies") or [] if c.get("isDefault")), None)
        return {"epic": epic, "name": inst.get("name"), "expiry": inst.get("expiry") or "-",
                "currency": currency or "GBP", "value_per_point": _f(inst.get("valueOfOnePip")),
                "one_pip_means": inst.get("onePipMeans"), "lot_size": _f(inst.get("lotSize")),
                "unit": inst.get("unit"), "margin_factor": _f(inst.get("marginFactor")),
                "min_size": _f((rules.get("minDealSize") or {}).get("value")),
                "market_status": (m.get("snapshot") or {}).get("marketStatus")}

    # ── prices ───────────────────────────────────────────────────────────
    def prices(self, epic: str, start: datetime | None = None, end: datetime | None = None,
               resolution: str = "HOUR", max_points: int | None = None) -> pd.DataFrame:
        """Candles between start and end, or (with max_points and no dates) the latest max_points
        candles. Either way it's ONE request; IG counts the candles against a weekly allowance."""
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {sorted(RESOLUTIONS)}")
        params: dict = {"resolution": resolution, "pageSize": 0}
        if start is not None:
            params["from"] = start.strftime("%Y-%m-%dT%H:%M:%S")
        if end is not None:
            params["to"] = end.strftime("%Y-%m-%dT%H:%M:%S")
        if max_points:
            params["max"] = int(max_points)
        body = self.request("GET", f"/prices/{epic}", 3, params=params)
        self.last_allowance = (body.get("metadata") or {}).get("allowance")
        return parse_prices(body)

    # ── account / positions (demo) ───────────────────────────────────────
    def accounts(self) -> list[dict]:
        return self.request("GET", "/accounts", 1).get("accounts", [])

    def positions(self) -> list[dict]:
        out = []
        for p in self.request("GET", "/positions", 2).get("positions", []):
            pos, mkt = p.get("position", {}), p.get("market", {})
            sign = 1 if pos.get("direction") == "BUY" else -1
            out.append({"deal_id": pos.get("dealId"), "epic": mkt.get("epic"), "size": sign * _f(pos.get("size") or 0),
                        "level": _f(pos.get("level")), "currency": pos.get("currency"),
                        "bid": _f(mkt.get("bid")), "offer": _f(mkt.get("offer"))})
        return out

    def net_position(self, epic: str) -> float:
        return sum(p["size"] for p in self.positions() if p["epic"] == epic)

    def place_market_order(self, epic: str, direction: str, size: float, expiry: str = "-",
                           currency: str = "GBP") -> dict:
        """Market order on the DEMO account. forceOpen=False nets against an opposite position,
        so buying 2 while short 1 leaves you long 1."""
        if direction not in ("BUY", "SELL") or size <= 0:
            raise ValueError("direction must be BUY/SELL and size > 0")
        ref = self.request("POST", "/positions/otc", 2, {
            "epic": epic, "expiry": expiry, "direction": direction, "size": round(float(size), 4),
            "orderType": "MARKET", "currencyCode": currency, "forceOpen": False, "guaranteedStop": False,
        }).get("dealReference")
        return self.confirm(ref)

    def confirm(self, deal_reference: str) -> dict:
        for _ in range(5):
            try:
                c = self.request("GET", f"/confirms/{deal_reference}", 1)
                return {"deal_reference": deal_reference, "status": c.get("dealStatus"), "reason": c.get("reason"),
                        "deal_id": c.get("dealId"), "level": _f(c.get("level")), "size": _f(c.get("size")),
                        "direction": c.get("direction")}
            except IGError:
                time.sleep(1)
        return {"deal_reference": deal_reference, "status": "UNKNOWN"}


def parse_prices(body: dict) -> pd.DataFrame:
    """IG /prices (v3) -> one row per candle with bid/ask OHLC, mid close and spread."""
    rows = []
    for p in body.get("prices", []):
        ts = p.get("snapshotTimeUTC") or p.get("snapshotTime")
        row = {"timestamp": pd.Timestamp(str(ts).replace("/", "-"), tz="UTC") if ts else None,
               "volume": _f(p.get("lastTradedVolume"))}
        for k, name in (("openPrice", "open"), ("highPrice", "high"), ("lowPrice", "low"), ("closePrice", "close")):
            q = p.get(k) or {}
            row[f"bid_{name}"], row[f"ask_{name}"] = _f(q.get("bid")), _f(q.get("ask"))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["timestamp"])
    df = df.dropna(subset=["timestamp", "bid_close", "ask_close"])
    df["mid_close"] = (df["bid_close"] + df["ask_close"]) / 2
    df["spread"] = df["ask_close"] - df["bid_close"]
    return df.sort_values("timestamp").reset_index(drop=True)


def _f(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _error_code(resp) -> str:
    try:
        return str(resp.json().get("errorCode") or resp.text[:200])
    except Exception:
        return resp.text[:200]
