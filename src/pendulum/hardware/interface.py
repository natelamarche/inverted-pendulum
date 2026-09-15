from __future__ import annotations

import time
import numpy as np
import serial

from .state_estimator import StateEstimator

PENDULUM_COUNTS_PER_DEGREE = 6.666667
ROTOR_MICROSTEPS_PER_DEGREE = 8.888889

class HardwarePendulum:
    def __init__(
        self, 
        port: str, 
        state_estimator: StateEstimator,
        baudrate: int = 115200,
        control_period: float  = 0.002,
    ):
        self.serial = serial.Serial(
            port,
            baudrate=baudrate,
            timeout=0,
        )
        
        self.state_estimator = state_estimator
        self.control_period = control_period
        
        self._rx_buffer = bytearray()
        self._measurement_time = 0.0
        
        self.serial.reset_input_buffer()

    def read_state(self, timeout: float = 0.1) -> np.ndarray:
        deadline = time.monotonic() + timeout
        latest_state = None

        while time.monotonic() < deadline:
            if self.serial.in_waiting:
                self._rx_buffer.extend(
                    self.serial.read(self.serial.in_waiting)
                )

            while b"\n" in self._rx_buffer:
                line, _, remainder = self._rx_buffer.partition(b"\n")
                self._rx_buffer = bytearray(remainder)

                state = self._parse_record(line.rstrip(b"\r"))

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

        raise TimeoutError("Timed out waiting for hardware telemetry")

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
    
    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()
            
    def __enter__(self) -> HardwarePendulum:
        return self
    
    def __exit__(self, *args: object) -> None:
        self.close()