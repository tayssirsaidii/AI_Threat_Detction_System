from sqlalchemy.orm import Session
from models import Report

def create_report(db: Session, threat_type: str, recommendation: str, model_performance: str):
    db_report = Report(
        threat_type=threat_type,
        recommendation=recommendation,
        model_performance=model_performance
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report
