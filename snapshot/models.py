# forest_app/persistence/models.py

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional # Ensure basic types are imported

# --- SQLAlchemy Imports ---
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, ForeignKey, JSON, Text # Keep JSON for potential fallback/other uses
)
# --- ADDED/MODIFIED IMPORT for JSONB ---
from sqlalchemy.dialects.postgresql import JSONB # Import specifically for PostgreSQL
# --- END ADDED/MODIFIED IMPORT ---
from sqlalchemy.orm import relationship, Mapped, mapped_column, DeclarativeBase
from sqlalchemy.sql import func # For server-side timestamp defaults

# --- Base Class ---
class Base(DeclarativeBase):
    pass

# --- User Model ---
class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # --- Relationships ---
    snapshots: Mapped[List["MemorySnapshotModel"]] = relationship("MemorySnapshotModel", back_populates="user")
    task_footprints: Mapped[List["TaskFootprintModel"]] = relationship("TaskFootprintModel", back_populates="user")
    reflection_logs: Mapped[List["ReflectionLogModel"]] = relationship("ReflectionLogModel", back_populates="user")


# --- Memory Snapshot Model ---
class MemorySnapshotModel(Base):
    __tablename__ = "memory_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    # --- MODIFIED Line Below: Changed JSON to JSONB ---
    snapshot_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True) # Use JSONB if using PostgreSQL
    # --- END MODIFICATION ---
    codename: Mapped[Optional[str]] = mapped_column(String, nullable=True) # Added codename field
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Consider if an updated_at is needed here too
    # updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


    # --- Relationships ---
    user: Mapped["UserModel"] = relationship("UserModel", back_populates="snapshots")


# --- Task Footprint Model ---
class TaskFootprintModel(Base):
    __tablename__ = "task_footprints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False) # e.g., 'issued', 'completed', 'failed', 'skipped'
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # --- Store snapshot_ref as JSON(B) ---
    snapshot_ref: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True) # Optional snapshot context at time of event
    # --- Store metadata as JSON(B) ---
    event_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True) # e.g., {"success": true/false, "reason": "..."}


    # --- Relationships ---
    user: Mapped["UserModel"] = relationship("UserModel", back_populates="task_footprints")


# --- Reflection Log Model ---
class ReflectionLogModel(Base):
    __tablename__ = "reflection_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    reflection_text: Mapped[str] = mapped_column(Text, nullable=False) # Use Text for potentially long reflections
    # --- Store snapshot_ref as JSON(B) ---
    snapshot_ref: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True) # Optional snapshot context at time of reflection
    # --- Store metadata as JSON(B) ---
    analysis_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True) # e.g., sentiment scores, patterns identified


    # --- Relationships ---
    user: Mapped["UserModel"] = relationship("UserModel", back_populates="reflection_logs")
