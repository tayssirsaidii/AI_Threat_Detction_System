from sqlalchemy import Column, Integer, String
from database import Base

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    threat_type = Column(String, index=True)
    recommendation = Column(String)
    model_performance = Column(String)
