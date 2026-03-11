import time
import os
from faculties.shahid_clock import ShahidClock, Moment

def test_moment_has_utc_and_hours():
    clock = ShahidClock(birth_file="tests/test_birth.txt")
    m = clock.now()
    assert isinstance(m, Moment)
    assert m.utc_iso.startswith("202")   # basic UTC ISO sanity
    assert isinstance(m.hours_online, float)
    assert m.hours_online >= 0.0

def test_birth_file_persists():
    if os.path.exists("tests/test_birth.txt"):
        os.remove("tests/test_birth.txt")
    c1 = ShahidClock(birth_file="tests/test_birth.txt")
    t1 = c1.now().hours_online
    time.sleep(0.05)
    c2 = ShahidClock(birth_file="tests/test_birth.txt")
    t2 = c2.now().hours_online
    assert t2 > t1   # second instance reads same birth, more time has passed

def test_hours_increases():
    clock = ShahidClock(birth_file="tests/test_birth2.txt")
    t1 = clock.now().hours_online
    time.sleep(0.05)
    t2 = clock.now().hours_online
    assert t2 > t1
