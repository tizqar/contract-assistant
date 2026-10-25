"""The tables of the web app: accounts, folders, contracts, decisions and the activity log.

Postgres is the database the app runs on (docker-compose.yml); the tests run the same tables
on SQLite, so the column types here are the portable ones with a Postgres variant.

A contract is a file a user keeps in a folder. Its analysis (the clause candidates the
extractor found) is stored once per text in `analyses` and shared by every contract with the
same text; the reader's decisions, chats, generated texts and activity belong to the contract.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.dialects.postgresql import BYTEA, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

ID_LENGTH = 80
EMAIL_LENGTH = 320
NAME_LENGTH = 255
DOC_ID_LENGTH = 80

ADMIN = "admin"
USER = "user"
ROLES = (ADMIN, USER)
VIEW = "view"
EDIT = "edit"
SHARE_LEVELS = (VIEW, EDIT)
CONFIRMED = "confirmed"
REJECTED = "rejected"
VERDICTS = (CONFIRMED, REJECTED)
IN_REVIEW = "in_review"
SUBMITTED = "submitted"
APPROVED = "approved"
REVIEW_STATES = (IN_REVIEW, SUBMITTED, APPROVED)
PDF = "pdf"
TXT = "txt"
KINDS = (PDF, TXT)

# Constraint and index names are fixed, so the migrations name them the same on every database.
NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def new_id() -> str:
    """Return a new random id: 32 hex characters."""
    return uuid.uuid4().hex


def utcnow() -> datetime:
    """Return the current time in UTC, with its time zone."""
    return datetime.now(tz=UTC)


class UTCDateTime(TypeDecorator):
    """A time stored with its time zone and always read back in UTC, SQLite included."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("a stored time must carry its time zone")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


JSONData = JSON().with_variant(JSONB(), "postgresql")
Bytes = LargeBinary().with_variant(BYTEA(), "postgresql")
# SQLite numbers rows only through an INTEGER primary key.
BigId = BigInteger().with_variant(Integer(), "sqlite")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(value) for value in values)})"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)
    type_annotation_map: ClassVar[dict] = {datetime: UTCDateTime(), dict[str, Any]: JSONData}


class User(Base):
    """An account: an admin or a user. Accounts are deactivated, never deleted."""

    __tablename__ = "users"
    __table_args__ = (CheckConstraint(_in("role", ROLES), name="role"),)

    id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True, default=new_id)
    # Kept in lower case; the sign-in form lowers what is typed.
    email: Mapped[str] = mapped_column(String(EMAIL_LENGTH), unique=True)
    name: Mapped[str] = mapped_column(String(NAME_LENGTH))
    password_hash: Mapped[str] = mapped_column(String(NAME_LENGTH))
    role: Mapped[str] = mapped_column(String(16), default=USER)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Set when an admin creates the account or resets its password.
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    last_login_at: Mapped[datetime | None]


class LoginSession(Base):
    """A signed-in browser. The cookie holds a random token; only its hash is stored."""

    __tablename__ = "sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    expires_at: Mapped[datetime]
    last_seen_at: Mapped[datetime] = mapped_column(default=utcnow)
    user_agent: Mapped[str | None] = mapped_column(String(NAME_LENGTH))
    ip: Mapped[str | None] = mapped_column(String(64))


class Folder(Base):
    """A folder in a user's space; parent None is the top of that space."""

    __tablename__ = "folders"

    id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("folders.id"), index=True)
    name: Mapped[str] = mapped_column(String(NAME_LENGTH))
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    # In the trash since then; everything inside counts as in the trash too.
    deleted_at: Mapped[datetime | None]
    deleted_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"))


class FolderShare(Base):
    """A folder shared with another user, with everything inside it, to view or to edit."""

    __tablename__ = "folder_shares"
    __table_args__ = (CheckConstraint(_in("level", SHARE_LEVELS), name="level"),)

    folder_id: Mapped[str] = mapped_column(
        ForeignKey("folders.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    level: Mapped[str] = mapped_column(String(8))
    granted_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    granted_at: Mapped[datetime] = mapped_column(default=utcnow)


class Analysis(Base):
    """The clause candidates of one text, as `store.DocumentCache` JSON; shared by contracts."""

    __tablename__ = "analyses"

    doc_id: Mapped[str] = mapped_column(String(DOC_ID_LENGTH), primary_key=True)
    data: Mapped[dict[str, Any]]
    extractor: Mapped[str] = mapped_column(String(NAME_LENGTH))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


class Blob(Base):
    """The bytes of an uploaded file, stored once per content."""

    __tablename__ = "blobs"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    media_type: Mapped[str] = mapped_column(String(100))
    size: Mapped[int] = mapped_column(BigInteger)
    data: Mapped[bytes] = mapped_column(Bytes)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Contract(Base):
    """A contract file in a folder: its name, its analysis, its file and its review state."""

    __tablename__ = "contracts"
    __table_args__ = (
        CheckConstraint(_in("kind", KINDS), name="kind"),
        CheckConstraint(_in("review_status", REVIEW_STATES), name="review_status"),
    )

    id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    # None: at the top of the owner's space.
    folder_id: Mapped[str | None] = mapped_column(ForeignKey("folders.id"), index=True)
    name: Mapped[str] = mapped_column(String(NAME_LENGTH))
    doc_id: Mapped[str] = mapped_column(ForeignKey("analyses.doc_id"), index=True)
    file_sha256: Mapped[str | None] = mapped_column(ForeignKey("blobs.sha256"))
    kind: Mapped[str] = mapped_column(String(4))
    review_status: Mapped[str] = mapped_column(String(16), default=IN_REVIEW)
    # What the admin wrote when sending the review back; cleared on the next submission.
    review_note: Mapped[str | None] = mapped_column(Text)
    review_changed_at: Mapped[datetime | None]
    review_changed_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    deleted_at: Mapped[datetime | None]
    deleted_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"))


class Decision(Base):
    """The current verdict on one clause type of one contract; its history is in `events`."""

    __tablename__ = "decisions"
    __table_args__ = (CheckConstraint(_in("verdict", VERDICTS), name="verdict"),)

    contract_id: Mapped[str] = mapped_column(
        ForeignKey("contracts.id", ondelete="CASCADE"), primary_key=True
    )
    clause_type: Mapped[str] = mapped_column(String(NAME_LENGTH), primary_key=True)
    verdict: Mapped[str] = mapped_column(String(10))
    start: Mapped[int] = mapped_column(Integer)
    end: Mapped[int] = mapped_column(Integer)
    # None for decisions imported from the cache folder, made before there were accounts.
    decided_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    decided_at: Mapped[datetime] = mapped_column(default=utcnow)


class Generated(Base):
    """A saved generation (summary, key terms, obligations, overview) of one contract."""

    __tablename__ = "generated"

    kind: Mapped[str] = mapped_column(String(40), primary_key=True)
    # The contract id and a digest of the model, thresholds and prompt version.
    key: Mapped[str] = mapped_column(String(160), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(ID_LENGTH), index=True)
    data: Mapped[dict[str, Any]]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Feedback(Base):
    """A thumbs up or down on a chat answer."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    contract_id: Mapped[str | None] = mapped_column(ForeignKey("contracts.id"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    question: Mapped[str] = mapped_column(Text)
    route_kind: Mapped[str] = mapped_column(String(40))
    clause_types: Mapped[list[str]] = mapped_column(JSONData)
    liked: Mapped[bool] = mapped_column(Boolean)
    comment: Mapped[str | None] = mapped_column(Text)
    at: Mapped[datetime] = mapped_column(default=utcnow)


class ChatMessage(Base):
    """One message of a user's chat about a contract; each user's chat is their own."""

    __tablename__ = "chat_messages"
    __table_args__ = (Index("ix_chat_messages_contract_user", "contract_id", "user_id"),)

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    contract_id: Mapped[str] = mapped_column(ForeignKey("contracts.id", ondelete="CASCADE"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    payload: Mapped[bytes] = mapped_column(Bytes)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Event(Base):
    """One entry of the activity log: who did what, when, and to what. Never changed."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    # None for what the system did, such as an import.
    actor_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(40), index=True)
    contract_id: Mapped[str | None] = mapped_column(ForeignKey("contracts.id"), index=True)
    folder_id: Mapped[str | None] = mapped_column(ForeignKey("folders.id"))
    subject_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    details: Mapped[dict[str, Any]] = mapped_column(JSONData, default=dict)
    ip: Mapped[str | None] = mapped_column(String(64))
