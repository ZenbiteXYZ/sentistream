import logging

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from sentistream.shared.config import settings


logger = logging.getLogger(settings.app.name)


# --- PostgreSQL Setup (SQLAlchemy Async) ---
try:
    engine = create_async_engine(
        settings.database.postgres_dsn,
        echo=(settings.app.env == "development"),
        pool_size=5,
        max_overflow=10,
    )
    AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)
    Base = declarative_base()
except Exception as e:
    logger.error(f"Failed to initialize PostgreSQL engine: {e}")
    raise


async def get_db_session():
    """Dependency to provide an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# --- Redis Setup ---
try:
    redis_client = Redis.from_url(
        settings.database.redis_url,
        decode_responses=True,  # Returns strings instead of bytes
    )
except Exception as e:
    logger.error(f"Failed to initialize Redis client: {e}")
    raise


async def ping_services():
    """Utility to test connections on startup."""
    try:
        # Import models here to ensure they are registered with Base.metadata before initialization
        import sentistream.shared.models  # noqa

        # Test Redis
        redis_client.ping()
        logger.info("Redis connected successfully.")

        # Test Postgres
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("PostgreSQL connected and metadata initialized.")

    except Exception as e:
        logger.error(f"Connection testing failed: {e}")
