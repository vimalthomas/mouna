"""initial schema: users, signs, attempts, sign_progress

Revision ID: 0001
Revises:
Create Date: 2026-07-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("avatar_url", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "signs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("gloss", sa.String(length=100), nullable=False),
    )
    op.create_index("ix_signs_gloss", "signs", ["gloss"], unique=True)

    op.create_table(
        "attempts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("sign_id", sa.Integer(), sa.ForeignKey("signs.id"), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_attempts_user_id", "attempts", ["user_id"])
    op.create_index("ix_attempts_sign_id", "attempts", ["sign_id"])
    op.create_index("ix_attempts_session_id", "attempts", ["session_id"])
    op.create_index("ix_attempts_created_at", "attempts", ["created_at"])

    op.create_table(
        "sign_progress",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("sign_id", sa.Integer(), sa.ForeignKey("signs.id"), nullable=False),
        sa.Column("total_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("correct_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "sign_id", name="uq_sign_progress_user_sign"),
    )
    op.create_index("ix_sign_progress_user_id", "sign_progress", ["user_id"])
    op.create_index("ix_sign_progress_sign_id", "sign_progress", ["sign_id"])


def downgrade() -> None:
    op.drop_table("sign_progress")
    op.drop_index("ix_attempts_created_at", table_name="attempts")
    op.drop_index("ix_attempts_session_id", table_name="attempts")
    op.drop_index("ix_attempts_sign_id", table_name="attempts")
    op.drop_index("ix_attempts_user_id", table_name="attempts")
    op.drop_table("attempts")
    op.drop_index("ix_signs_gloss", table_name="signs")
    op.drop_table("signs")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
