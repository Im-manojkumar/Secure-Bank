"""Time-based one-time codes (TOTP, RFC 6238), the 6-digit codes authenticator apps show.

Written out here rather than taken from a library because it is short: the
code is an HMAC-SHA1 of the current 30-second time step, keyed with a secret
shared with the user's app once, at setup.
"""

import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote

STEP_SECONDS = 30
DIGITS = 6
# Accept the step before and after too, for clocks a little out of sync.
DRIFT_STEPS = 1
ISSUER = "SecureBank"


def new_secret():
    """A random 160-bit secret in base32, the form authenticator apps accept."""
    return base64.b32encode(secrets.token_bytes(20)).decode()


def code_at(secret, step):
    key = base64.b32decode(secret)
    digest = hmac.new(key, struct.pack(">Q", step), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    number = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(number % 10 ** DIGITS).zfill(DIGITS)


def current_step(now=None):
    return int((time.time() if now is None else now) // STEP_SECONDS)


def matching_step(secret, code, last_used_step=None, now=None):
    """The time step this code belongs to, or None if it is wrong, expired or already used.

    Refusing a step at or before last_used_step stops a code seen over someone's
    shoulder from being used again.
    """
    code = str(code or "").strip().replace(" ", "")
    if len(code) != DIGITS or not code.isdigit():
        return None
    now_step = current_step(now)
    for step in range(now_step - DRIFT_STEPS, now_step + DRIFT_STEPS + 1):
        if last_used_step is not None and step <= last_used_step:
            continue
        if hmac.compare_digest(code_at(secret, step), code):
            return step
    return None


def setup_uri(secret, email):
    """The otpauth:// link an authenticator app reads (typed in, or as a QR code)."""
    return (f"otpauth://totp/{quote(ISSUER)}:{quote(email)}?secret={secret}"
            f"&issuer={quote(ISSUER)}&digits={DIGITS}&period={STEP_SECONDS}")
