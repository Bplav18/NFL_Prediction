"""
Central configuration loaded from environment variables.
Import `settings` anywhere in the codebase instead of calling os.environ directly.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    NFL_DATA_API_KEY: str = os.getenv("NFL_DATA_API_KEY", "")
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")

    ENV: str = os.getenv("ENV", "development")


settings = Settings()
