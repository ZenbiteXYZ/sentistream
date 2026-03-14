from sqlalchemy import JSON, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from sentistream.shared.db import Base


class ReviewRecord(Base):
    __tablename__ = "reviews"

    id = Column(String, primary_key=True, index=True)
    text = Column(String, nullable=False)
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    metadata_col = Column(JSON, nullable=True)

    # Machine Learning Data
    cluster_id = Column(Integer, index=True, nullable=False)
    cluster_name = Column(String, index=True, nullable=True)

    # Coordinates for visualization (Storing as JSON arrays for simplicity in Postgres)
    reduced_coords = Column(JSON, nullable=False)

    # We store the 384D float embedding as compressed binary or JSON.
    # For now, we store as JSON array.
    full_embedding = Column(JSON, nullable=False)
