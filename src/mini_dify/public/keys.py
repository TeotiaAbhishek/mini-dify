import hashlib
import secrets

KEY_PREFIX = "mdk-"  # "mini-dify key": easy to spot in logs and in secret scanners


def generate_key() -> str:
    """A new random key: the prefix plus 32 URL-safe characters (192 bits of randomness)."""
    return KEY_PREFIX + secrets.token_urlsafe(24)


def hash_key(key: str) -> str:
    """What we store instead of the key. SHA-256 is enough because keys are long and
    random: there is no dictionary of likely keys to try, unlike with passwords."""
    return hashlib.sha256(key.encode()).hexdigest()


def display_prefix(key: str) -> str:
    """The start of a key, safe to show: "mdk-Ab12Cd34"."""
    return key[: len(KEY_PREFIX) + 8]
