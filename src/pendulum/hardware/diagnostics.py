from dataclasses import dataclass

def _milli(value: str) -> float:
    parsed = int(value)
    return float("nan") if parsed == -2147483648 else parsed / 1000



@dataclass(frozen=True)
class FirmwareDiagnostic:
    raw: str
    active: bool
    stopped: bool
    reason: str
    sequence: int | None
    age_ms: int
    acceleration: float
    speed: float
    rotor_steps: int
    stop_tick_ms: int
    applied_acceleration: float | None = None
    speed_limited: bool | None = None

    @classmethod
    def parse(cls, raw: str) -> "FirmwareDiagnostic":
        fields = raw.split(",")
        if fields[:2] == ["EVENT", "STOP"]:
            fields = fields[2:]
        elif fields[0] == "STATUS":
            fields = fields[1:]
        else:
            raise ValueError("Unknown diagnostic record")
        if not fields or (fields[0], len(fields)) not in {("1", 11), ("2", 13)}:
            raise ValueError("Unsupported diagnostic version or field count")
        applied, limited = None, None
        if fields[0] == "2":
            if fields[12] not in {"0", "1"}:
                raise ValueError("Invalid limiting flag")
            applied, limited = _milli(fields[11]), fields[12] == "1"
        _, active, stopped, reason, have_seq, seq, age, acc, speed, rotor, tick = fields[:11]
        if any(flag not in {"0", "1"} for flag in (active, stopped, have_seq)):
            raise ValueError("Invalid diagnostic flags")
        sequence, age_ms, stop_tick_ms = int(seq), int(age), int(tick)
        if any(value < 0 or value > 0xFFFFFFFF
               for value in (sequence, age_ms, stop_tick_ms)):
            raise ValueError("Invalid unsigned diagnostic field")
        return cls(
            raw, active == "1", stopped == "1", reason,
            sequence if have_seq == "1" else None,
            age_ms, _milli(acc), _milli(speed), int(rotor), stop_tick_ms,
            applied, limited,
        )


class FirmwareStoppedError(RuntimeError):
    def __init__(self, diagnostic: FirmwareDiagnostic):
        self.diagnostic = diagnostic
        super().__init__(
            f"Firmware stopped: {diagnostic.reason}; "
            f"last accepted sequence={diagnostic.sequence}, "
            f"action age={diagnostic.age_ms} ms, "
            f"requested acceleration={diagnostic.acceleration:g} rad/s², "
            f"applied acceleration={diagnostic.applied_acceleration} rad/s², "
            f"speed limiting={diagnostic.speed_limited}, "
            f"internal speed={diagnostic.speed:g} rad/s, "
            f"rotor={diagnostic.rotor_steps} microsteps, "
            f"stop tick={diagnostic.stop_tick_ms} ms. "
            f"Raw diagnostic: {diagnostic.raw}"
        )
