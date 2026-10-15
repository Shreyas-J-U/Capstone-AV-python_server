import math
import struct
import unittest
from pathlib import Path

from protocol.constants import SensorType, SensorFormat
from protocol.models import Observation, Sensor, Termination
from perception.safe_distance import SafeDistanceCalculator, VehiclePrior, SafetyEvaluation
from perception.sensor_utils import extract_ego_speed
from perception.track_history import TrackHistory


class TestSafeDistanceCalculator(unittest.TestCase):

    def setUp(self):
        # Resolve config path relative to repository root
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.config_path = repo_root / "configs" / "perception.yaml"
        self.calculator = SafeDistanceCalculator(config_path=self.config_path)

    def test_initialization_and_config(self):
        """Test default config loading and geometry initialization."""
        self.assertAlmostEqual(self.calculator.image_width, 1280)
        self.assertAlmostEqual(self.calculator.image_height, 720)
        self.assertAlmostEqual(self.calculator.fov_horizontal_deg, 90.0)
        self.assertAlmostEqual(self.calculator.focal_length_px, 640.0)
        self.assertAlmostEqual(self.calculator.standstill_distance_m, 2.5)
        self.assertAlmostEqual(self.calculator.time_headway_s, 1.8)

        # Check vehicle priors loaded
        car_prior = self.calculator.get_prior("car")
        truck_prior = self.calculator.get_prior("truck")
        person_prior = self.calculator.get_prior("person")

        self.assertAlmostEqual(car_prior.height_m, 1.50)
        self.assertAlmostEqual(truck_prior.height_m, 3.40)
        self.assertAlmostEqual(person_prior.height_m, 1.70)
        self.assertGreater(truck_prior.alpha, car_prior.alpha)
        self.assertGreater(person_prior.alpha, car_prior.alpha)

    def test_pinhole_distance_estimation(self):
        """Test monocular distance estimation scales inversely with bbox height."""
        # Car with height 150px vs height 75px
        dist_close = self.calculator.estimate_distance([100, 100, 250, 250], "car")  # h = 150px
        dist_far = self.calculator.estimate_distance([100, 100, 200, 175], "car")    # h = 75px

        # Z = (f * H) / h -> Halving pixel height should double estimated distance
        self.assertAlmostEqual(dist_close, (640.0 * 1.5) / 150.0, places=2)
        self.assertAlmostEqual(dist_far, (640.0 * 1.5) / 75.0, places=2)
        self.assertAlmostEqual(dist_far, dist_close * 2.0, places=2)

    def test_vehicle_type_effect_on_safe_distance(self):
        """
        Verify that vehicle type influences safe distance:
        Truck and Vulnerable Road Users (pedestrians) require higher safe distance than a car at the same speed.
        """
        ego_speed = 15.0  # 15 m/s (~54 km/h)

        d_car, _, _ = self.calculator.compute_safe_distance(ego_speed_ms=ego_speed, class_name="car")
        d_truck, _, _ = self.calculator.compute_safe_distance(ego_speed_ms=ego_speed, class_name="truck")
        d_bus, _, _ = self.calculator.compute_safe_distance(ego_speed_ms=ego_speed, class_name="bus")
        d_person, _, _ = self.calculator.compute_safe_distance(ego_speed_ms=ego_speed, class_name="person")

        self.assertGreater(d_truck, d_car, "Truck should have greater safe distance than Car")
        self.assertGreater(d_bus, d_car, "Bus should have greater safe distance than Car")
        self.assertGreater(d_person, d_car, "Pedestrian should have greater safe distance than Car")

    def test_size_categorization(self):
        """Verify size classification and size scaling factors."""
        cat_car, factor_car = self.calculator.categorize_size("car", [100, 100, 250, 220])
        cat_truck, factor_truck = self.calculator.categorize_size("truck", [100, 100, 500, 300])
        cat_person, factor_person = self.calculator.categorize_size("person", [100, 100, 130, 200])

        self.assertEqual(cat_car, "LARGE")
        self.assertEqual(cat_truck, "EXTRA_LARGE")
        self.assertEqual(cat_person, "SMALL")
        self.assertGreater(factor_truck, factor_car)

    def test_ego_speed_scaling(self):
        """Verify safe distance increases monotonically with ego vehicle speed."""
        d_0, _, _ = self.calculator.compute_safe_distance(0.0, "car")
        d_10, _, _ = self.calculator.compute_safe_distance(10.0, "car")
        d_25, _, _ = self.calculator.compute_safe_distance(25.0, "car")

        self.assertAlmostEqual(d_0, self.calculator.standstill_distance_m, places=1)
        self.assertGreater(d_10, d_0)
        self.assertGreater(d_25, d_10)

    def test_safety_status_transitions(self):
        """Verify SAFE, WARNING, and CRITICAL status thresholds."""
        ego_speed = 10.0  # 10 m/s -> safe distance approx 2.5 + 10*1.8*1.0 = 20.5 m

        # In-path centered bbox [590, 100, 690, 200] (center cx=640)
        # Safe: target is 40m away
        eval_safe = self.calculator.evaluate_safety(
            bbox=[590, 100, 690, 200],
            class_name="car",
            ego_speed_ms=ego_speed,
            direct_distance_m=40.0,
        )
        self.assertEqual(eval_safe.safety_status, "SAFE")
        self.assertTrue(eval_safe.in_ego_path)
        self.assertGreater(eval_safe.distance_margin_m, 0.0)

        # Warning: target is within 1.25 * d_safe (e.g. 23m)
        eval_warning = self.calculator.evaluate_safety(
            bbox=[590, 100, 690, 200],
            class_name="car",
            ego_speed_ms=ego_speed,
            direct_distance_m=22.0,
        )
        self.assertEqual(eval_warning.safety_status, "WARNING")

        # Critical: target is inside d_safe (e.g. 10m)
        eval_critical = self.calculator.evaluate_safety(
            bbox=[590, 100, 690, 200],
            class_name="car",
            ego_speed_ms=ego_speed,
            direct_distance_m=10.0,
        )
        self.assertEqual(eval_critical.safety_status, "CRITICAL")
        self.assertLess(eval_critical.distance_margin_m, 0.0)

        # Off-path target in adjacent lane should be SAFE
        eval_adj = self.calculator.evaluate_safety(
            bbox=[100, 100, 200, 200],
            class_name="car",
            ego_speed_ms=ego_speed,
            direct_distance_m=10.0,
        )
        self.assertFalse(eval_adj.in_ego_path)
        self.assertTrue("SAFE" in eval_adj.safety_status)

    def test_enrich_detection(self):
        """Verify detection dictionary enrichment."""
        detection = {
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.92,
            "bbox": [590.0, 100.0, 690.0, 200.0],
        }

        enriched = self.calculator.enrich_detection(detection, ego_speed_ms=12.0)

        expected_keys = [
            "class_id", "class_name", "confidence", "bbox",
            "bbox_dimensions", "size_category", "size_factor",
            "estimated_distance_m", "safe_distance_m",
            "safe_distance_thw_m", "safe_distance_rss_m",
            "distance_margin_m", "safety_status",
            "lateral_offset_m", "lane_position", "in_ego_path",
        ]
        for key in expected_keys:
            self.assertIn(key, enriched, f"Missing key '{key}' in enriched detection")

        self.assertIsInstance(enriched["safe_distance_m"], float)
        self.assertIsInstance(enriched["estimated_distance_m"], float)
        self.assertTrue(any(s in enriched["safety_status"] for s in ["SAFE", "WARNING", "CRITICAL"]))

    def test_sensor_utils_speed_extraction(self):
        """Test extraction of speed from Observation sensors."""
        # 1. Direct SPEED sensor (float32)
        obs_speed = Observation(
            episode_id=1, step_id=0, frame_id=1, simulation_time=0.0, image=None,
            sensors=[
                Sensor(
                    type=SensorType.SPEED,
                    format=SensorFormat.FLOAT32,
                    value=struct.pack("<f", 14.2),
                )
            ],
            reward=0.0, termination=Termination(terminated=False),
        )
        speed = extract_ego_speed(obs_speed)
        self.assertAlmostEqual(speed, 14.2, places=2)

        # 2. VELOCITY vector sensor (vx=12.0, vy=5.0, vz=0.0 -> speed = 13.0)
        obs_vel = Observation(
            episode_id=1, step_id=0, frame_id=1, simulation_time=0.0, image=None,
            sensors=[
                Sensor(
                    type=SensorType.VELOCITY,
                    format=SensorFormat.VECTOR3,
                    value=struct.pack("<fff", 12.0, 5.0, 0.0),
                )
            ],
            reward=0.0, termination=Termination(terminated=False),
        )
        speed_vel = extract_ego_speed(obs_vel)
        self.assertAlmostEqual(speed_vel, 13.0, places=2)

        # 3. None or missing sensors
        self.assertEqual(extract_ego_speed(None), 0.0)

    def test_track_history_distance_tracking(self):
        """Verify TrackHistory retains distance history and safe distance metadata."""
        history = TrackHistory(max_history=10)

        obj_frame1 = {
            "track_id": 1,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.9,
            "bbox": [100, 100, 200, 200],
            "estimated_distance_m": 25.0,
            "safe_distance_m": 15.0,
            "safety_status": "SAFE",
        }
        obj_frame2 = {
            "track_id": 1,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.9,
            "bbox": [102, 100, 202, 200],
            "estimated_distance_m": 23.5,
            "safe_distance_m": 15.0,
            "safety_status": "SAFE",
        }

        history.update([obj_frame1])
        history.update([obj_frame2])

        dist_hist = history.get_distance_history(1)
        self.assertEqual(dist_hist, [25.0, 23.5])

        meta = history.get_metadata(1)
        self.assertEqual(meta["safety_status"], "SAFE")
        self.assertEqual(meta["estimated_distance_m"], 23.5)
        self.assertEqual(meta["safe_distance_m"], 15.0)


if __name__ == "__main__":
    unittest.main()
