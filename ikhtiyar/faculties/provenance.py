from dataclasses import dataclass
from faculties.shahid_clock import Moment


VALID_ORIGIN_TYPES = {"ayah", "inference", "human", "feed"}


@dataclass
class ProvenanceRecord:
    origin_type: str       # ayah | inference | human | feed
    origin_ref: str        # e.g. "2:164" or "bilal" or "feed:reuters"
    session_id: str
    moment: Moment
    confidence: float      # 0.0 - 1.0

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if self.origin_type not in VALID_ORIGIN_TYPES:
            raise ValueError(
                f"origin_type must be one of {sorted(VALID_ORIGIN_TYPES)}, "
                f"got {self.origin_type!r}"
            )


def attach(edge_props: dict, record: ProvenanceRecord) -> dict:
    """Merge provenance into a Neo4j edge properties dict."""
    return {
        **edge_props,
        "prov_origin_type": record.origin_type,
        "prov_origin_ref": record.origin_ref,
        "prov_session_id": record.session_id,
        "prov_utc": record.moment.utc_iso,
        "prov_hours_online": record.moment.hours_online,
        "prov_confidence": record.confidence,
    }
