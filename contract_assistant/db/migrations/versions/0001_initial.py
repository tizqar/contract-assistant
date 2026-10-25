"""Accounts, folders, contracts, decisions and the activity log

Revision ID: 0001
Revises:
Create Date: 2026-10-25 15:26:21.661678
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0001'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('analyses',
    sa.Column('doc_id', sa.String(length=80), nullable=False),
    sa.Column('data', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('extractor', sa.String(length=255), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('doc_id', name=op.f('pk_analyses'))
    )
    op.create_table('blobs',
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('media_type', sa.String(length=100), nullable=False),
    sa.Column('size', sa.BigInteger(), nullable=False),
    sa.Column('data', sa.LargeBinary().with_variant(postgresql.BYTEA(), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('sha256', name=op.f('pk_blobs'))
    )
    op.create_table('generated',
    sa.Column('kind', sa.String(length=40), nullable=False),
    sa.Column('key', sa.String(length=160), nullable=False),
    sa.Column('doc_id', sa.String(length=80), nullable=False),
    sa.Column('data', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('kind', 'key', name=op.f('pk_generated'))
    )
    op.create_index(op.f('ix_generated_doc_id'), 'generated', ['doc_id'], unique=False)
    op.create_table('users',
    sa.Column('id', sa.String(length=80), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('active', sa.Boolean(), nullable=False),
    sa.Column('must_change_password', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("role IN ('admin', 'user')", name=op.f('ck_users_role')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('email', name=op.f('uq_users_email'))
    )
    op.create_table('folders',
    sa.Column('id', sa.String(length=80), nullable=False),
    sa.Column('owner_id', sa.String(length=80), nullable=False),
    sa.Column('parent_id', sa.String(length=80), nullable=True),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('created_by', sa.String(length=80), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('deleted_by', sa.String(length=80), nullable=True),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_folders_created_by_users')),
    sa.ForeignKeyConstraint(['deleted_by'], ['users.id'], name=op.f('fk_folders_deleted_by_users')),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_folders_owner_id_users')),
    sa.ForeignKeyConstraint(['parent_id'], ['folders.id'], name=op.f('fk_folders_parent_id_folders')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_folders'))
    )
    op.create_index(op.f('ix_folders_owner_id'), 'folders', ['owner_id'], unique=False)
    op.create_index(op.f('ix_folders_parent_id'), 'folders', ['parent_id'], unique=False)
    op.create_table('sessions',
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('user_id', sa.String(length=80), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('user_agent', sa.String(length=255), nullable=True),
    sa.Column('ip', sa.String(length=64), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_sessions_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('token_hash', name=op.f('pk_sessions'))
    )
    op.create_index(op.f('ix_sessions_user_id'), 'sessions', ['user_id'], unique=False)
    op.create_table('contracts',
    sa.Column('id', sa.String(length=80), nullable=False),
    sa.Column('owner_id', sa.String(length=80), nullable=False),
    sa.Column('folder_id', sa.String(length=80), nullable=True),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('doc_id', sa.String(length=80), nullable=False),
    sa.Column('file_sha256', sa.String(length=64), nullable=True),
    sa.Column('kind', sa.String(length=4), nullable=False),
    sa.Column('review_status', sa.String(length=16), nullable=False),
    sa.Column('review_note', sa.Text(), nullable=True),
    sa.Column('review_changed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('review_changed_by', sa.String(length=80), nullable=True),
    sa.Column('created_by', sa.String(length=80), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('deleted_by', sa.String(length=80), nullable=True),
    sa.CheckConstraint("kind IN ('pdf', 'txt')", name=op.f('ck_contracts_kind')),
    sa.CheckConstraint("review_status IN ('in_review', 'submitted', 'approved')", name=op.f('ck_contracts_review_status')),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_contracts_created_by_users')),
    sa.ForeignKeyConstraint(['deleted_by'], ['users.id'], name=op.f('fk_contracts_deleted_by_users')),
    sa.ForeignKeyConstraint(['doc_id'], ['analyses.doc_id'], name=op.f('fk_contracts_doc_id_analyses')),
    sa.ForeignKeyConstraint(['file_sha256'], ['blobs.sha256'], name=op.f('fk_contracts_file_sha256_blobs')),
    sa.ForeignKeyConstraint(['folder_id'], ['folders.id'], name=op.f('fk_contracts_folder_id_folders')),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_contracts_owner_id_users')),
    sa.ForeignKeyConstraint(['review_changed_by'], ['users.id'], name=op.f('fk_contracts_review_changed_by_users')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_contracts'))
    )
    op.create_index(op.f('ix_contracts_doc_id'), 'contracts', ['doc_id'], unique=False)
    op.create_index(op.f('ix_contracts_folder_id'), 'contracts', ['folder_id'], unique=False)
    op.create_index(op.f('ix_contracts_owner_id'), 'contracts', ['owner_id'], unique=False)
    op.create_table('folder_shares',
    sa.Column('folder_id', sa.String(length=80), nullable=False),
    sa.Column('user_id', sa.String(length=80), nullable=False),
    sa.Column('level', sa.String(length=8), nullable=False),
    sa.Column('granted_by', sa.String(length=80), nullable=False),
    sa.Column('granted_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("level IN ('view', 'edit')", name=op.f('ck_folder_shares_level')),
    sa.ForeignKeyConstraint(['folder_id'], ['folders.id'], name=op.f('fk_folder_shares_folder_id_folders'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['granted_by'], ['users.id'], name=op.f('fk_folder_shares_granted_by_users')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_folder_shares_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('folder_id', 'user_id', name=op.f('pk_folder_shares'))
    )
    op.create_index(op.f('ix_folder_shares_user_id'), 'folder_shares', ['user_id'], unique=False)
    op.create_table('chat_messages',
    sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False),
    sa.Column('contract_id', sa.String(length=80), nullable=False),
    sa.Column('user_id', sa.String(length=80), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.Column('role', sa.String(length=16), nullable=False),
    sa.Column('payload', sa.LargeBinary().with_variant(postgresql.BYTEA(), 'postgresql'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['contract_id'], ['contracts.id'], name=op.f('fk_chat_messages_contract_id_contracts'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_chat_messages_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_chat_messages'))
    )
    op.create_index('ix_chat_messages_contract_user', 'chat_messages', ['contract_id', 'user_id'], unique=False)
    op.create_table('decisions',
    sa.Column('contract_id', sa.String(length=80), nullable=False),
    sa.Column('clause_type', sa.String(length=255), nullable=False),
    sa.Column('verdict', sa.String(length=10), nullable=False),
    sa.Column('start', sa.Integer(), nullable=False),
    sa.Column('end', sa.Integer(), nullable=False),
    sa.Column('decided_by', sa.String(length=80), nullable=True),
    sa.Column('decided_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("verdict IN ('confirmed', 'rejected')", name=op.f('ck_decisions_verdict')),
    sa.ForeignKeyConstraint(['contract_id'], ['contracts.id'], name=op.f('fk_decisions_contract_id_contracts'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['decided_by'], ['users.id'], name=op.f('fk_decisions_decided_by_users')),
    sa.PrimaryKeyConstraint('contract_id', 'clause_type', name=op.f('pk_decisions'))
    )
    op.create_table('events',
    sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False),
    sa.Column('at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('actor_id', sa.String(length=80), nullable=True),
    sa.Column('action', sa.String(length=40), nullable=False),
    sa.Column('contract_id', sa.String(length=80), nullable=True),
    sa.Column('folder_id', sa.String(length=80), nullable=True),
    sa.Column('subject_user_id', sa.String(length=80), nullable=True),
    sa.Column('details', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('ip', sa.String(length=64), nullable=True),
    sa.ForeignKeyConstraint(['actor_id'], ['users.id'], name=op.f('fk_events_actor_id_users')),
    sa.ForeignKeyConstraint(['contract_id'], ['contracts.id'], name=op.f('fk_events_contract_id_contracts')),
    sa.ForeignKeyConstraint(['folder_id'], ['folders.id'], name=op.f('fk_events_folder_id_folders')),
    sa.ForeignKeyConstraint(['subject_user_id'], ['users.id'], name=op.f('fk_events_subject_user_id_users')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_events'))
    )
    op.create_index(op.f('ix_events_action'), 'events', ['action'], unique=False)
    op.create_index(op.f('ix_events_actor_id'), 'events', ['actor_id'], unique=False)
    op.create_index(op.f('ix_events_at'), 'events', ['at'], unique=False)
    op.create_index(op.f('ix_events_contract_id'), 'events', ['contract_id'], unique=False)
    op.create_table('feedback',
    sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False),
    sa.Column('contract_id', sa.String(length=80), nullable=True),
    sa.Column('user_id', sa.String(length=80), nullable=True),
    sa.Column('question', sa.Text(), nullable=False),
    sa.Column('route_kind', sa.String(length=40), nullable=False),
    sa.Column('clause_types', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('liked', sa.Boolean(), nullable=False),
    sa.Column('comment', sa.Text(), nullable=True),
    sa.Column('at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['contract_id'], ['contracts.id'], name=op.f('fk_feedback_contract_id_contracts')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_feedback_user_id_users')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_feedback'))
    )
    op.create_index(op.f('ix_feedback_contract_id'), 'feedback', ['contract_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_feedback_contract_id'), table_name='feedback')
    op.drop_table('feedback')
    op.drop_index(op.f('ix_events_contract_id'), table_name='events')
    op.drop_index(op.f('ix_events_at'), table_name='events')
    op.drop_index(op.f('ix_events_actor_id'), table_name='events')
    op.drop_index(op.f('ix_events_action'), table_name='events')
    op.drop_table('events')
    op.drop_table('decisions')
    op.drop_index('ix_chat_messages_contract_user', table_name='chat_messages')
    op.drop_table('chat_messages')
    op.drop_index(op.f('ix_folder_shares_user_id'), table_name='folder_shares')
    op.drop_table('folder_shares')
    op.drop_index(op.f('ix_contracts_owner_id'), table_name='contracts')
    op.drop_index(op.f('ix_contracts_folder_id'), table_name='contracts')
    op.drop_index(op.f('ix_contracts_doc_id'), table_name='contracts')
    op.drop_table('contracts')
    op.drop_index(op.f('ix_sessions_user_id'), table_name='sessions')
    op.drop_table('sessions')
    op.drop_index(op.f('ix_folders_parent_id'), table_name='folders')
    op.drop_index(op.f('ix_folders_owner_id'), table_name='folders')
    op.drop_table('folders')
    op.drop_table('users')
    op.drop_index(op.f('ix_generated_doc_id'), table_name='generated')
    op.drop_table('generated')
    op.drop_table('blobs')
    op.drop_table('analyses')
