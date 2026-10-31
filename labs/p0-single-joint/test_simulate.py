"""Run with: python -m unittest discover -s labs/p0-single-joint -p 'test_*.py'"""
import csv
import tempfile
import unittest
from pathlib import Path

from simulate import Config, clamp, save_csv, simulate


class JointSimulationTests(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp(2.0, 0.8), 0.8)
        self.assertEqual(clamp(-2.0, 0.8), -0.8)

    def test_reaches_target(self):
        rows, result = simulate(Config())
        self.assertEqual(len(rows), 2000)
        self.assertLess(abs(result["final_error_rad"]), 0.01)
        self.assertLessEqual(result["peak_command_torque_Nm"], 0.8 + 1e-12)

    def test_no_controller_no_motion(self):
        rows, result = simulate(Config(kp=0, kd=0))
        self.assertEqual(result["final_angle_rad"], 0.0)
        self.assertTrue(all(r["torque_Nm"] == 0 for r in rows))

    def test_disturbance_changes_response(self):
        _, nominal = simulate(Config())
        _, disturbed = simulate(Config(disturbance=-0.2))
        self.assertNotEqual(nominal["final_angle_rad"], disturbed["final_angle_rad"])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            simulate(Config(inertia=0))
        with self.assertRaises(ValueError):
            simulate(Config(dt=0.03))

    def test_csv_has_header_and_samples(self):
        rows, _ = simulate(Config(duration=0.01))
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "samples.csv"
            save_csv(rows, output)
            with output.open(encoding="utf-8", newline="") as fp:
                values = list(csv.DictReader(fp))
            self.assertEqual(len(values), len(rows))
            self.assertAlmostEqual(float(values[-1]["time_s"]), 0.01)


if __name__ == "__main__":
    unittest.main()
