"""Dependency-free single-joint rigid-body PD control simulation.

Model: J * angular_acceleration = commanded_torque - b * angular_velocity
                                      + external_torque.
Educational simulation only. Not suitable for real motor control.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    inertia: float = 0.02       # kg*m^2
    damping: float = 0.04       # N*m*s/rad
    kp: float = 0.8            # N*m/rad
    kd: float = 0.3            # N*m*s/rad
    torque_limit: float = 0.8   # N*m
    target: float = 1.0        # rad
    duration: float = 4.0      # seconds
    dt: float = 0.002          # seconds
    disturbance: float = 0.0  # external torque (N*m), t in [2.0, 2.3)

    def validate(self) -> None:
        if self.inertia <= 0 or self.torque_limit <= 0 or self.dt <= 0 or self.duration <= 0:
            raise ValueError("inertia, torque_limit, dt and duration must be > 0")
        if self.damping < 0 or self.kp < 0 or self.kd < 0:
            raise ValueError("damping, kp and kd must be >= 0")
        if self.dt > 0.02:
            raise ValueError("dt > 0.02 s: use a smaller step for this teaching example")


def clamp(value: float, magnitude: float) -> float:
    return max(-magnitude, min(magnitude, value))


def simulate(config: Config) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Integrate a 1-DOF rigid joint using semi-implicit Euler."""
    config.validate()
    theta, omega = 0.0, 0.0
    effort = 0.0
    peak_torque = 0.0
    rows: list[dict[str, float]] = []
    steps = int(round(config.duration / config.dt))
    for i in range(steps):
        t = i * config.dt
        tau = clamp(config.kp * (config.target - theta) - config.kd * omega,
                    config.torque_limit)
        external = config.disturbance if 2.0 <= t < 2.3 else 0.0
        accel = (tau - config.damping * omega + external) / config.inertia
        omega += accel * config.dt
        theta += omega * config.dt
        # Integral of absolute mechanical power: an effort measure, NOT battery energy.
        effort += abs(tau * omega) * config.dt
        peak_torque = max(peak_torque, abs(tau))
        rows.append({"time_s": (i + 1) * config.dt,
                     "angle_rad": theta,
                     "velocity_rad_s": omega,
                     "torque_Nm": tau,
                     "external_Nm": external})
    return rows, {"final_angle_rad": theta,
                  "final_error_rad": config.target - theta,
                  "peak_command_torque_Nm": peak_torque,
                  "abs_mechanical_work_J": effort}


def save_csv(rows: list[dict[str, float]], filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["time_s", "angle_rad",
                                                  "velocity_rad_s", "torque_Nm", "external_Nm"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 virtual joint: PD control simulation")
    parser.add_argument("--target", type=float, default=1.0, help="target joint angle, rad")
    parser.add_argument("--kp", type=float, default=0.8, help="position proportional gain")
    parser.add_argument("--kd", type=float, default=0.3, help="velocity derivative gain")
    parser.add_argument("--disturbance", type=float, default=0.0,
                        help="external torque applied from 2.0 to 2.3 s, N*m")
    parser.add_argument("--csv", type=Path, default=None,
                        help="optional path to export samples as CSV")
    args = parser.parse_args()
    config = Config(target=args.target, kp=args.kp, kd=args.kd,
                    disturbance=args.disturbance)
    rows, result = simulate(config)
    for label, value in result.items():
        print(f"{label}: {value:.6f}")
    if args.csv is not None:
        save_csv(rows, args.csv)
        print(f"Wrote {len(rows)} samples to: {args.csv}")


if __name__ == "__main__":
    main()
