import logging

from sqlalchemy.orm import Session

from db.models import Sign

logger = logging.getLogger(__name__)


def sync_signs(db: Session, glosses: list[str]) -> None:
    existing = {row.gloss for row in db.query(Sign.gloss).all()}
    new_glosses = [g for g in glosses if g not in existing]
    for gloss in new_glosses:
        db.add(Sign(gloss=gloss))
    if new_glosses:
        db.commit()
        logger.info("Seeded %d new sign(s) into the database", len(new_glosses))


def get_or_create_sign(db: Session, gloss: str) -> Sign:
    sign = db.query(Sign).filter(Sign.gloss == gloss).first()
    if sign is None:
        sign = Sign(gloss=gloss)
        db.add(sign)
        db.commit()
        db.refresh(sign)
    return sign
