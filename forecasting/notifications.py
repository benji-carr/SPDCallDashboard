from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Mapping


LOGGER = logging.getLogger(__name__)


def _env_flag(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _split_recipients(value: str) -> list[str]:
    recipients = [item.strip() for item in value.split(",")]
    return [item for item in recipients if item]


@dataclass(frozen=True)
class SmtpNotificationConfig:
    enabled: bool
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_from: str
    smtp_to: tuple[str, ...]
    smtp_starttls: bool = True
    subject_prefix: str = "[SPD XGBoost]"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SmtpNotificationConfig | None:
        env = env or os.environ
        if not _env_flag(env.get("SPD_XGBOOST_NOTIFICATION_ENABLED")):
            return None
        required = {
            "SPD_XGBOOST_SMTP_HOST": env.get("SPD_XGBOOST_SMTP_HOST"),
            "SPD_XGBOOST_SMTP_PORT": env.get("SPD_XGBOOST_SMTP_PORT"),
            "SPD_XGBOOST_SMTP_USERNAME": env.get("SPD_XGBOOST_SMTP_USERNAME"),
            "SPD_XGBOOST_SMTP_PASSWORD": env.get("SPD_XGBOOST_SMTP_PASSWORD"),
            "SPD_XGBOOST_EMAIL_FROM": env.get("SPD_XGBOOST_EMAIL_FROM"),
            "SPD_XGBOOST_EMAIL_TO": env.get("SPD_XGBOOST_EMAIL_TO"),
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                "SPD XGBoost email notifications are enabled but missing required environment variables: "
                f"{names}"
            )
        recipients = _split_recipients(required["SPD_XGBOOST_EMAIL_TO"] or "")
        if not recipients:
            raise ValueError(
                "SPD XGBoost email notifications are enabled but SPD_XGBOOST_EMAIL_TO did not contain any recipients."
            )
        try:
            smtp_port = int(required["SPD_XGBOOST_SMTP_PORT"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SPD XGBoost email notifications are enabled but SPD_XGBOOST_SMTP_PORT is not a valid integer."
            ) from exc
        return cls(
            enabled=True,
            smtp_host=(required["SPD_XGBOOST_SMTP_HOST"] or "").strip(),
            smtp_port=smtp_port,
            smtp_username=(required["SPD_XGBOOST_SMTP_USERNAME"] or "").strip(),
            smtp_password=required["SPD_XGBOOST_SMTP_PASSWORD"] or "",
            smtp_from=(required["SPD_XGBOOST_EMAIL_FROM"] or "").strip(),
            smtp_to=tuple(recipients),
            smtp_starttls=_env_flag(env.get("SPD_XGBOOST_SMTP_STARTTLS", "1")),
            subject_prefix=(env.get("SPD_XGBOOST_EMAIL_SUBJECT_PREFIX") or "[SPD XGBoost]").strip() or "[SPD XGBoost]",
            timeout_seconds=float(env.get("SPD_XGBOOST_SMTP_TIMEOUT_SECONDS", "30")),
        )

    def secret_values(self) -> tuple[str, ...]:
        return tuple(value for value in [self.smtp_password] if value)


class SmtpEmailTransport:
    def __init__(self, config: SmtpNotificationConfig, *, smtp_factory=smtplib.SMTP):
        self.config = config
        self.smtp_factory = smtp_factory

    def send_email(self, *, subject: str, body: str) -> None:
        message = EmailMessage()
        message["Subject"] = f"{self.config.subject_prefix} {subject}".strip()
        message["From"] = self.config.smtp_from
        message["To"] = ", ".join(self.config.smtp_to)
        message.set_content(body)
        with self.smtp_factory(
            self.config.smtp_host,
            self.config.smtp_port,
            timeout=self.config.timeout_seconds,
        ) as smtp:
            smtp.ehlo()
            if self.config.smtp_starttls:
                smtp.starttls()
                smtp.ehlo()
            smtp.login(self.config.smtp_username, self.config.smtp_password)
            smtp.send_message(message)


def collect_secret_values(
    env: Mapping[str, str] | None = None,
    *,
    config: SmtpNotificationConfig | None = None,
) -> tuple[str, ...]:
    env = env or os.environ
    secret_names = [
        name
        for name in env
        if any(token in name.upper() for token in ("PASSWORD", "SECRET", "TOKEN"))
    ]
    secrets = [env[name] for name in secret_names if env.get(name)]
    if config is not None:
        secrets.extend(config.secret_values())
    return tuple(sorted({value for value in secrets if value}, key=len, reverse=True))


def redact_secrets(text: str, secret_values: tuple[str, ...]) -> str:
    redacted = str(text)
    for value in secret_values:
        redacted = redacted.replace(value, "***REDACTED***")
    return redacted
