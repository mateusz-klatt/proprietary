from collections.abc import Sequence
from datetime import datetime
from datetime import timezone
from sqlalchemy import text
from alembic import op
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None
API_KEYS = [
    ("polygon_api_key", "IvLGKVNdIYx1qvhdA8KLPKnx_UJjtgt9", "api", "Polygon.io API key"),
    ("walutomat_api_key", "dlj9xfnjjj6kwmh1ozxkc1tn9", "api", "Walutomat API key"),
    (
        "kraken_api_key",
        "mKVV6ZfXVnoQXjrwXDjtYpUAZlSfPD/MWmPgjK+/vjd9wvT6Qpw5FNyJ",
        "api",
        "Kraken API key",
    ),
    (
        "kraken_api_secret",
        "/YxPQO/+oR8gEO0/7ljtxPvkw/NYgAhzHPfhAWt5ilcjHsMx2UHN8Du8JFMDruhCzqPsTnNM5/dO61Z8yilhMw==",
        "api",
        "Kraken API secret",
    ),
    ("zonda_api_key", "9a78ae4f-4908-451c-a132-deeee9645e65", "api", "Zonda API key"),
    ("zonda_api_secret", "1ce9fce8-53f7-400b-92d6-c124e991bf38", "api", "Zonda API secret"),
]
def upgrade() -> None:
    conn = op.get_bind()
    now = datetime.now(tz=timezone.utc)
    for setting in API_KEYS:
        conn.execute(
            text(
                """
                INSERT INTO settings (key, value, category, description, is_encrypted, updated_at)
                VALUES (:key, :value, :category, :description, 0, :updated_at)
                """
            ),
            {
                "key": setting[0],
                "value": setting[1],
                "category": setting[2],
                "description": setting[3],
                "updated_at": now,
            },
        )
def downgrade() -> None:
    conn = op.get_bind()
    for setting in API_KEYS:
        conn.execute(text("DELETE FROM settings WHERE key = :key"), {"key": setting[0]})
