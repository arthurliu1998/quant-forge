"""Secure Telegram notification sender. Never logs bot token or full API URL. Supports sensitivity levels."""
import logging
import re
import requests

from trade_forge.signals.engine import Signal

logger = logging.getLogger(__name__)


def format_signal_message(signal: Signal, sensitivity: str = "medium") -> str:
    if sensitivity == "high":
        direction = signal.direction.upper() if signal.direction else "SIGNAL"
        return f"{signal.symbol}: {direction} signal detected [{signal.priority}]"
    if sensitivity == "medium":
        msg = f"[{signal.priority}] {signal.symbol}: {signal.type}\n{signal.message}"
        msg = re.sub(r'\$[\d,]+\.?\d*', '', msg)
        return msg.strip()
    # low — full details
    return (f"[{signal.priority}] {signal.symbol}\nSignal: {signal.type}\n"
            f"{signal.message}\nDirection: {signal.direction}")


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, sensitivity: str = "medium"):
        self._token = token
        self._chat_id = chat_id
        self.sensitivity = sensitivity

    def is_configured(self) -> bool:
        return bool(self._token and self._chat_id)

    def send_signal(self, signal: Signal) -> bool:
        if not self.is_configured():
            logger.warning("Telegram not configured — skipping notification")
            return False
        msg = format_signal_message(signal, self.sensitivity)
        return self._send_text(msg)

    def send_text(self, text: str) -> bool:
        if not self.is_configured():
            return False
        return self._send_text(text)

    def _send_text(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        try:
            resp = requests.post(url, json={"chat_id": self._chat_id, "text": text}, timeout=10)
            if resp.status_code == 200:
                return True
            logger.warning("Telegram send failed: HTTP %d", resp.status_code)
            return False
        except Exception as e:
            logger.warning("Telegram send failed: %s", type(e).__name__)
            return False
