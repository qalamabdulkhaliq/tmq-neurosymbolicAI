from dataclasses import dataclass
from datetime import datetime, timezone
import os


@dataclass
class Moment:
    utc_iso: str
    hours_online: float


class ShahidClock:
    def __init__(self, birth_file: str = "shahid_birth.txt"):
        self.birth_file = birth_file
        if os.path.exists(birth_file):
            with open(birth_file) as f:
                self._birth = datetime.fromisoformat(f.read().strip())
        else:
            self._birth = datetime.now(timezone.utc)
            with open(birth_file, "w") as f:
                f.write(self._birth.isoformat())

    def now(self) -> Moment:
        now = datetime.now(timezone.utc)
        hours = (now - self._birth).total_seconds() / 3600.0
        return Moment(utc_iso=now.isoformat(), hours_online=round(hours, 6))
