"""v5: dominant hand, practice stories, sentence attempts

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("users") as b:
        b.add_column(sa.Column("dominant_hand", sa.String(length=5), nullable=False, server_default="right"))
    op.create_table(
        "practice_stories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_practice_stories_user_id", "practice_stories", ["user_id"])
    op.create_index("ix_practice_stories_created_at", "practice_stories", ["created_at"])
    op.create_table(
        "sentence_attempts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("story_id", sa.Integer(), sa.ForeignKey("practice_stories.id"), nullable=True),
        sa.Column("english", sa.Text(), nullable=False),
        sa.Column("gloss", sa.String(length=500), nullable=False),
        sa.Column("results", sa.Text(), nullable=False),
        sa.Column("signs_good", sa.Integer(), nullable=False),
        sa.Column("signs_total", sa.Integer(), nullable=False),
        sa.Column("all_good", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_sentence_attempts_user_id", "sentence_attempts", ["user_id"])
    op.create_index("ix_sentence_attempts_created_at", "sentence_attempts", ["created_at"])


def downgrade() -> None:
    op.drop_table("sentence_attempts")
    op.drop_table("practice_stories")
    with op.batch_alter_table("users") as b:
        b.drop_column("dominant_hand")
