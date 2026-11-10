"""Start a local Postgres 16 + pgvector without Docker (uses the `pgserver` wheel).

  python scripts/dev_postgres.py          # prints a DATABASE_URL; server keeps running
  python scripts/dev_postgres.py --stop

Data lives in .pgdata/ (git-ignored). Prefer `docker compose up db` where Docker is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pgserver

DATA = Path(__file__).resolve().parents[1] / ".pgdata"


def main() -> None:
    srv = pgserver.get_server(str(DATA), cleanup_mode=None)
    if "--stop" in sys.argv:
        srv.cleanup()
        print("stopped")
        return
    srv.psql("CREATE EXTENSION IF NOT EXISTS vector;")
    print(srv.get_uri())


if __name__ == "__main__":
    main()
