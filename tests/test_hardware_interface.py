from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import serial

from pendulum.hardware.interface import MAX_LINE_BYTES, HardwarePendulum
from pendulum.hardware.state_estimator import StateEstimator


class FakeSerial:
    def __init__(self):
        self.is_open = True
        self.rx = bytearray()
        self.commands = []
        self.on_write = Mock()

    @property
    def in_waiting(self):
        return len(self.rx)

    def reset_input_buffer(self):
        self.rx.clear()

    def read(self, size):
        data = bytes(self.rx[:size])
        del self.rx[:size]
        return data

    def write(self, data):
        self.commands.append(data)
        self.on_write()
        return len(data)

    def close(self):
        self.is_open = False


@pytest.fixture
def hardware(monkeypatch):
    connection = FakeSerial()
    monkeypatch.setattr(
        "pendulum.hardware.interface.serial.Serial",
        lambda *args, **kwargs: connection,
    )
    return HardwarePendulum("fake", StateEstimator(velocity_smoothing=0))


def test_close_stops_after_start_was_sent_but_write_failed(hardware):
    hardware._zeroed = True
    hardware.serial.on_write.side_effect = [OSError("drain failed"), None]

    with pytest.raises(OSError, match="drain failed"), hardware:
        hardware.start(10)

    assert hardware.serial.commands == [b"START,1", b"STOP\r"]
    assert not hardware.serial.is_open


@pytest.mark.parametrize("command", ["ZERO", "STOP", "START,1,-10.000000", "A,27,-10.000000"])
def test_command_chunks_are_paced(hardware, monkeypatch, command):
    events = []
    hardware.serial.on_write.side_effect = lambda: events.append(
        ("write", hardware.serial.commands[-1])
    )
    monkeypatch.setattr(
        "pendulum.hardware.interface.time.sleep",
        lambda delay: events.append(("sleep", delay)),
    )

    hardware._write_command(command)

    payload = f"{command}\r".encode("ascii")
    assert b"".join(hardware.serial.commands) == payload
    assert all(1 <= len(chunk) <= 7 for chunk in hardware.serial.commands)
    assert events == [
        event
        for chunk in hardware.serial.commands
        for event in [("write", chunk), ("sleep", 0.001)]
    ]


def test_concurrent_commands_do_not_interleave(hardware):
    commands = [f"A,{sequence},-10.000000" for sequence in range(20)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(hardware._write_command, commands))

    received = b"".join(hardware.serial.commands).decode("ascii").split("\r")
    assert received[-1] == ""
    assert sorted(received[:-1]) == sorted(commands)


@pytest.mark.parametrize("original", [RuntimeError("control failed"), KeyboardInterrupt()])
def test_cleanup_failure_preserves_original_exception(hardware, original):
    hardware.serial.on_write.side_effect = serial.SerialTimeoutException("Write timeout")

    with pytest.raises(type(original)) as raised, hardware:
        raise original

    assert raised.value is original
    assert "cleanup also failed" in original.__notes__[0]
    assert hardware.serial.commands == [b"STOP\r"]
    assert not hardware.serial.is_open


def test_short_write_mid_command_stops_sending_and_invalidates_session(hardware):
    hardware._running = hardware._zeroed = True
    hardware.serial.write = Mock(side_effect=[7, 1])

    with pytest.raises(OSError, match="Incomplete serial command"):
        hardware.apply_acceleration(-10)

    assert hardware.serial.write.call_count == 2
    assert not hardware._running
    assert not hardware._zeroed


def test_close_stops_even_without_start_and_is_idempotent(hardware):
    hardware.close()
    hardware.close()

    assert hardware.serial.commands == [b"STOP\r"]
    assert not hardware.serial.is_open


def test_close_closes_port_even_when_stop_fails(hardware):
    hardware.serial.on_write.side_effect = OSError("drain failed")

    with pytest.raises(OSError, match="drain failed"):
        hardware.close()

    assert hardware.serial.commands == [b"STOP\r"]
    assert not hardware.serial.is_open


def test_zero_discards_queued_telemetry_and_resets_estimator(hardware):
    old_record = b"2 0 0 600 800 0 0 0 0\n"
    hardware.serial.rx.extend(old_record)
    hardware.read_state()
    hardware.serial.rx.extend(old_record)
    hardware._rx_buffer.extend(old_record)
    hardware.serial.on_write.side_effect = lambda: hardware.serial.rx.extend(
        old_record + b"ACK,ZERO\r\n" + b"2 0 0 0 0 0 0 0 0\n"
    )

    hardware.zero()

    # The sample received alongside ACK survives and initializes the estimator.
    np.testing.assert_array_equal(hardware.read_state(), np.zeros(4))

    hardware.serial.rx.extend(b"2 0 0 0 0 0 0 0 0\n")
    np.testing.assert_array_equal(hardware.read_state(), np.zeros(4))
    assert hardware.serial.commands == [b"ZERO\r"]


def test_zero_handles_fragmented_ack_and_partial_following_telemetry(hardware):
    chunks = iter([b"ACK,ZE", b"RO\r", b"\n2 0 0 0"])
    original_read = hardware.serial.read

    def read_fragment(size):
        data = original_read(size)
        hardware.serial.rx.extend(next(chunks, b""))
        return data

    hardware.serial.read = read_fragment
    hardware.serial.on_write.side_effect = lambda: hardware.serial.rx.extend(b"noise\n")
    hardware.zero()

    hardware.serial.rx.extend(b" 0 0 0 0 0\n")
    np.testing.assert_array_equal(hardware.read_state(), np.zeros(4))


@pytest.mark.parametrize("response", [b"", b"ACK,OTHER\r\n", b"ACK,ZERO\r"])
def test_zero_times_out_without_complete_ack_and_preserves_estimator(hardware, response):
    hardware.state_estimator.reset = Mock()
    hardware._measurement_time = 1.0
    hardware._sequence = 7
    # An already queued acknowledgement must not satisfy the new request.
    hardware.serial.rx.extend(b"ACK,ZERO\r\n")
    hardware._rx_buffer.extend(b"ACK,ZERO\r\n")
    hardware.serial.on_write.side_effect = lambda: hardware.serial.rx.extend(response)

    with pytest.raises(TimeoutError, match="ACK,ZERO"):
        hardware.zero(timeout=0.001)

    hardware.state_estimator.reset.assert_not_called()
    assert hardware._measurement_time == 1.0
    assert hardware._sequence == 7


def test_zero_handles_firmware_running_rejection(hardware):
    hardware.state_estimator.reset = Mock()
    hardware.serial.on_write.side_effect = lambda: hardware.serial.rx.extend(
        b"ERR,ZERO,RUNNING\r\n"
    )

    with pytest.raises(RuntimeError, match="Firmware rejected ZERO"):
        hardware.zero()

    hardware.state_estimator.reset.assert_not_called()
    assert hardware._running


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_zero_rejects_invalid_timeout_before_sending(hardware, timeout):
    with pytest.raises(ValueError, match="Timeout"):
        hardware.zero(timeout=timeout)

    assert hardware.serial.commands == []


@pytest.mark.parametrize("stop_fails", [False, True])
def test_telemetry_timeout_stops_and_invalidates_session(hardware, stop_fails):
    hardware._running = hardware._zeroed = True
    failure = OSError("STOP failed")
    if stop_fails:
        hardware.serial.on_write.side_effect = failure

    with pytest.raises(TimeoutError, match="hardware telemetry") as raised:
        hardware.read_state(timeout=0.001)

    assert hardware.serial.commands == [b"STOP\r"]
    assert raised.value.__cause__ is (failure if stop_fails else None)
    assert not hardware._running
    assert not hardware._zeroed
    with pytest.raises(RuntimeError):
        hardware.apply_acceleration(1)
    with pytest.raises(RuntimeError):
        hardware.start()


@pytest.mark.parametrize("operation", ["read", "waiting", "write", "short_write", "reset"])
def test_serial_failure_invalidates_session(hardware, monkeypatch, operation):
    hardware._running = hardware._zeroed = True
    failure = OSError("serial failed")
    if operation == "read":
        hardware.serial.rx.extend(b"x")
        hardware.serial.read = Mock(side_effect=failure)
        action = hardware.read_state
    elif operation == "waiting":
        monkeypatch.setattr(FakeSerial, "in_waiting", property(Mock(side_effect=failure)))
        action = hardware.read_state
    elif operation == "reset":
        hardware._running = False
        hardware.serial.reset_input_buffer = Mock(side_effect=failure)
        action = hardware.zero
    else:
        if operation == "write":
            hardware.serial.write = Mock(side_effect=failure)
        else:
            hardware.serial.write = Mock(return_value=1)
        action = lambda: hardware.apply_acceleration(1)

    with pytest.raises(OSError):
        action()
    assert not hardware._running
    assert not hardware._zeroed


def test_oversized_line_discards_valid_looking_suffix_across_calls(hardware):
    hardware.serial.rx.extend(b"x" * (MAX_LINE_BYTES * 4))
    with pytest.raises(TimeoutError):
        hardware.read_state(timeout=0.02)
    assert len(hardware._rx_buffer) <= MAX_LINE_BYTES
    assert hardware._discarding_line

    # This telemetry-shaped suffix still belongs to the oversized record.
    hardware.serial.rx.extend(b"2 0 0 600 800 0 0 0 0\n")
    with pytest.raises(TimeoutError):
        hardware.read_state(timeout=0.002)

    hardware.serial.rx.extend(b"2 0 0 0 0 0 0 0 0\n")
    np.testing.assert_array_equal(hardware.read_state(), np.zeros(4))


def test_zero_discards_oversized_ack_suffix_and_accepts_next_line(hardware):
    hardware.serial.on_write.side_effect = lambda: hardware.serial.rx.extend(
        b"x" * (MAX_LINE_BYTES + 1) + b"ACK,ZERO\r\n"
    )
    with pytest.raises(TimeoutError, match="ACK,ZERO"):
        hardware.zero(timeout=0.01)
    assert not hardware._zeroed

    hardware.serial.on_write.side_effect = lambda: hardware.serial.rx.extend(
        b"x" * (MAX_LINE_BYTES * 2) + b"\nACK,ZERO\r\n"
        b"2 0 0 0 0 0 0 0 0\n"
    )
    hardware.zero()
    assert hardware._zeroed
    np.testing.assert_array_equal(hardware.read_state(), np.zeros(4))
