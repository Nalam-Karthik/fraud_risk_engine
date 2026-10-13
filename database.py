# database.py
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

DATABASE_URL = "sqlite:///./audit_vault.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ForensicAudit(Base):
    __tablename__ = "forensic_audits"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    verdict = Column(String, index=True)
    risk_probability = Column(Float)
    top_risk_drivers = Column(JSON)  # Stores top SHAP factors
    raw_payload = Column(JSON)

# Create tables
Base.metadata.create_all(bind=engine)