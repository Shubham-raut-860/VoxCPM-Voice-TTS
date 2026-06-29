import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Voice(Base):
    __tablename__ = "voices"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    owner = Column(String, nullable=True)
    file_path = Column(String, nullable=False)
    transcript = Column(Text, nullable=True)
    sample_rate = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    lora_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    generations = relationship("Generation", back_populates="voice", cascade="all, delete-orphan")

    @hybrid_property
    def has_lora(self) -> bool:
        return self.lora_path is not None


class Generation(Base):
    __tablename__ = "generations"

    id = Column(String, primary_key=True, default=_uuid)
    voice_id = Column(String, ForeignKey("voices.id"), nullable=False)
    text = Column(Text, nullable=False)
    file_path = Column(String, nullable=False)
    cfg_value = Column(Float, nullable=False)
    inference_timesteps = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    used_lora = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    voice = relationship("Voice", back_populates="generations")
