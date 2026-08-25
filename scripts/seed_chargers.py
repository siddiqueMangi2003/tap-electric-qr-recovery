from app.core.config import get_settings
from app.db.base import Base
from app.db.models import Charger
from app.db.session import build_engine, build_session_factory

SEED_CHARGERS = [
    Charger(
        charger_id="NL-TAP-E12345",
        qr_payload="https://tap-electric.com/c/NL-TAP-E12345",
        latitude=52.3676,
        longitude=4.9041,
    ),
    Charger(
        charger_id="NL-TAP-E67890",
        qr_payload="https://tap-electric.com/c/NL-TAP-E67890",
        latitude=52.3680,
        longitude=4.9036,
    ),
]


def main() -> None:
    settings = get_settings()
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    session = build_session_factory(engine)()
    try:
        for charger in SEED_CHARGERS:
            session.merge(charger)
        session.commit()
        print(f"Seeded {len(SEED_CHARGERS)} chargers")
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    main()
