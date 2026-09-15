from __future__ import annotations

import time
import numpy as np
import serial

from .state_estimator import StateEstimator

PENDULUM_COUNTS_PER_DEGREE = 6.666667
ROTOR_MICROSTEPS_PER_DEGREE = 8.888889
MAX_LINE_BYTES = 1024

class HardwarePendulum:
    def __init__(
        self, 
        port: str, 
        state_estimator: StateEstimator,
        baudrate: int = 230400,
        control_period: float  = 0.002,
    ):
        self.serial = serial.Serial(
            port,
            baudrate=baudrate,
            timeout=0,
            write_timeout=0.02,
        )
        
        self.state_estimator = state_estimator
        self.control_period = control_period
        
        self._rx_buffer = bytearray()
        self._discarding_line = False
        self._measurement_time = 0.0
        
        self._sequence = 0
        self._running = False
        self._zeroed = False
        
        self.serial.reset_input_buffer()

    def read_state(self, timeout: float = 0.1) -> np.ndarray:
        deadline = time.monotonic() + timeout
        latest_state = None

        while time.monotonic() < deadline:
            while time.monotonic() < deadline:
                line = self._read_line()
                if line is None:
                    break
                state = self._parse_record(line)

                if state is not None:
                    latest_state = state

            if latest_state is not None:
                return latest_state

            time.sleep(
                min(
                    0.0005,
                    max(0.0, deadline - time.monotonic()),
                )
            )

        error = TimeoutError("Timed out waiting for hardware telemetry")
        try:
            if self._running:
                self.stop()
        except Exception as stop_error:
            raise error from stop_error
        finally:
            self._invalidate_session()
        raise error

    def _invalidate_session(self) -> None:
        self._running = False
        self._zeroed = False

    def _read_line(self) -> bytes | None:
        read_once = False
        while True:
            newline = self._rx_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._rx_buffer[:newline])
                del self._rx_buffer[:newline + 1]
                if self._discarding_line or len(line) > MAX_LINE_BYTES:
                    self._discarding_line = False
                    continue
                return line.rstrip(b"\r")

            if self._discarding_line or len(self._rx_buffer) > MAX_LINE_BYTES:
                self._rx_buffer.clear()
                self._discarding_line = True

            if read_once:
                return None
            try:
                size = min(
                    self.serial.in_waiting,
                    MAX_LINE_BYTES + 1 - len(self._rx_buffer),
                )
                if not size:
                    return None
                self._rx_buffer.extend(self.serial.read(size))
            except (OSError, serial.SerialException):
                self._invalidate_session()
                raise
            read_once = True

    def _parse_record(self, line: bytes) -> np.ndarray | None:
        try:
            fields = line.decode("ascii").split()
        except UnicodeDecodeError:
            return None

        if len(fields) != 9 or fields[0] != "2":
            return None

        try:
            pendulum_counts = int(fields[3])
            rotor_microsteps = int(fields[4])
        except ValueError:
            return None

        theta = np.deg2rad(
            pendulum_counts / PENDULUM_COUNTS_PER_DEGREE
        )

        phi = np.deg2rad(
            rotor_microsteps / ROTOR_MICROSTEPS_PER_DEGREE
        )

        self._measurement_time += self.control_period

        return self.state_estimator.update(
            phi=phi,
            theta=theta,
            timestamp=self._measurement_time,
        )
    
    def zero(self, timeout: float = 0.1) -> None:
        if self._running:
            raise RuntimeError("Cannot zero pendulum while motor is running")

        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("Timeout must be finite and positive")

        self._zeroed = False
        try:
            self.serial.reset_input_buffer()
        except (OSError, serial.SerialException):
            self._invalidate_session()
            raise
        self._rx_buffer.clear()
        self._discarding_line = False
        self._write_command("ZERO")
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            while time.monotonic() < deadline:
                line = self._read_line()
                if line is None:
                    break
                if line == b"ACK,ZERO":
                    self._measurement_time = 0.0
                    self.state_estimator.reset()
                    self._sequence = 0
                    self._zeroed = True
                    return

                if line == b"ERR,ZERO,RUNNING":
                    self._running = True
                    raise RuntimeError("Firmware rejected ZERO: motor is running")

            time.sleep(min(0.0005, max(0.0, deadline - time.monotonic())))

        raise TimeoutError("Timed out waiting for ACK,ZERO")
    
    def start(self, acceleration: float = 0.0) -> None:
        if self._running:
            raise RuntimeError("Pendulum is already running")

        if not self._zeroed:
            raise RuntimeError("Call zero() successfully before starting")
        
        if not np.isfinite(acceleration):
            raise ValueError("Acceleration must be finite")
        
        if abs(acceleration) > 100.0:
            raise ValueError("Acceleration must be within ±100 rad/s²")

        self._sequence = 1

        self._write_command(
            f"START,{self._sequence},{acceleration:.6f}"
        )

        self._running = True

    def apply_acceleration(self, acceleration: float) -> None:
        if not self._running:
            raise RuntimeError("Pendulum has not been started")

        if not np.isfinite(acceleration):
            self.stop()
            raise ValueError("Acceleration must be finite")

        if abs(acceleration) > 100.0:
            self.stop()
            raise ValueError("Acceleration must be within ±100 rad/s²")
        
        self._sequence += 1

        self._write_command(
            f"A,{self._sequence},{acceleration:.6f}"
        )
    
    def stop(self) -> None:
        try:
            if self.serial.is_open:
                self._write_command("STOP")
        finally:
            self._running = False
            self._zeroed = False
        
    def _write_command(self, command: str) -> None:
        data = f"{command}\r".encode("ascii")
        try:
            if self.serial.write(data) != len(data):
                raise IOError("Incomplete serial command")
        except (OSError, serial.SerialException):
            self._invalidate_session()
            raise
    
    def close(self) -> None:
        if not self.serial.is_open:
            return
        
        try:
            self.stop()
        finally:
            self.serial.close()
            
    def __enter__(self) -> HardwarePendulum:
        return self
    
    def __exit__(self, *args: object) -> None:
        self.close()
