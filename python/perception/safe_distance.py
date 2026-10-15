import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml


@dataclass
class VehiclePrior:
    """Prior physical dimensions and safety parameters for an object class."""
    height_m: float
    width_m: float
    length_m: float
    alpha: float
    max_decel_ms2: float


@dataclass
class SafetyEvaluation:
    """Safe distance evaluation results for a single detected object."""
    bbox_dimensions: Dict[str, float]
    size_category: str
    size_factor: float
    estimated_distance_m: float
    safe_distance_m: float
    safe_distance_thw_m: float
    safe_distance_rss_m: float
    distance_margin_m: float
    safety_status: str
    lateral_offset_m: float = 0.0
    lane_position: str = "IN_PATH"
    in_ego_path: bool = True
    time_to_collision_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox_dimensions": self.bbox_dimensions,
            "size_category": self.size_category,
            "size_factor": round(self.size_factor, 3),
            "estimated_distance_m": round(self.estimated_distance_m, 2),
            "safe_distance_m": round(self.safe_distance_m, 2),
            "safe_distance_thw_m": round(self.safe_distance_thw_m, 2),
            "safe_distance_rss_m": round(self.safe_distance_rss_m, 2),
            "distance_margin_m": round(self.distance_margin_m, 2),
            "safety_status": self.safety_status,
            "lateral_offset_m": round(self.lateral_offset_m, 2),
            "lane_position": self.lane_position,
            "in_ego_path": self.in_ego_path,
            "time_to_collision_s": round(self.time_to_collision_s, 2) if self.time_to_collision_s is not None else None,
        }


class SafeDistanceCalculator:
    """
    Computes dynamic safe distance and monocular 3D distance/lateral position
    based on vehicle type, physical size, and driving corridor.
    """

    DEFAULT_PRIORS: Dict[str, VehiclePrior] = {
        "car": VehiclePrior(height_m=1.50, width_m=1.80, length_m=4.50, alpha=1.00, max_decel_ms2=7.0),
        "truck": VehiclePrior(height_m=3.40, width_m=2.50, length_m=10.00, alpha=1.45, max_decel_ms2=4.5),
        "bus": VehiclePrior(height_m=3.20, width_m=2.55, length_m=12.00, alpha=1.40, max_decel_ms2=4.8),
        "motorcycle": VehiclePrior(height_m=1.40, width_m=0.80, length_m=2.10, alpha=1.30, max_decel_ms2=5.5),
        "bicycle": VehiclePrior(height_m=1.40, width_m=0.60, length_m=1.70, alpha=1.35, max_decel_ms2=5.0),
        "person": VehiclePrior(height_m=1.70, width_m=0.50, length_m=0.30, alpha=1.60, max_decel_ms2=4.0),
        "traffic light": VehiclePrior(height_m=0.80, width_m=0.40, length_m=0.40, alpha=1.00, max_decel_ms2=0.0),
        "stop sign": VehiclePrior(height_m=0.75, width_m=0.75, length_m=0.10, alpha=1.00, max_decel_ms2=0.0),
        "default": VehiclePrior(height_m=1.50, width_m=1.80, length_m=4.50, alpha=1.00, max_decel_ms2=6.0),
    }

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        image_width: int = 1280,
        image_height: int = 720,
        fov_horizontal_deg: float = 90.0,
        standstill_distance_m: float = 2.5,
        time_headway_s: float = 1.8,
        reaction_time_s: float = 0.5,
        ego_comfortable_decel_ms2: float = 6.0,
        warning_margin_multiplier: float = 1.25,
        critical_margin_multiplier: float = 1.00,
        lane_width_m: float = 3.6,
        lateral_corridor_half_width_m: float = 1.85,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.fov_horizontal_deg = fov_horizontal_deg
        self.standstill_distance_m = standstill_distance_m
        self.time_headway_s = time_headway_s
        self.reaction_time_s = reaction_time_s
        self.ego_comfortable_decel_ms2 = ego_comfortable_decel_ms2
        self.warning_margin_multiplier = warning_margin_multiplier
        self.critical_margin_multiplier = critical_margin_multiplier
        self.lane_width_m = lane_width_m
        self.lateral_corridor_half_width_m = lateral_corridor_half_width_m

        self.priors: Dict[str, VehiclePrior] = dict(self.DEFAULT_PRIORS)

        # Load YAML config if provided and exists
        if config_path:
            self.load_config(config_path)

        # Compute focal length in pixels (fx = fy assuming square pixels)
        self.focal_length_px = (self.image_width / 2.0) / math.tan(
            math.radians(self.fov_horizontal_deg / 2.0)
        )

    def load_config(self, config_path: str | Path) -> None:
        """Load parameters and vehicle priors from a YAML configuration file."""
        path = Path(config_path)
        if not path.is_file():
            return

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            return

        cam = cfg.get("camera", {})
        if "image_width" in cam:
            self.image_width = int(cam["image_width"])
        if "image_height" in cam:
            self.image_height = int(cam["image_height"])
        if "fov_horizontal_deg" in cam:
            self.fov_horizontal_deg = float(cam["fov_horizontal_deg"])

        safety = cfg.get("safety_params", {})
        if "standstill_distance_m" in safety:
            self.standstill_distance_m = float(safety["standstill_distance_m"])
        if "time_headway_s" in safety:
            self.time_headway_s = float(safety["time_headway_s"])
        if "reaction_time_s" in safety:
            self.reaction_time_s = float(safety["reaction_time_s"])
        if "ego_comfortable_decel_ms2" in safety:
            self.ego_comfortable_decel_ms2 = float(safety["ego_comfortable_decel_ms2"])
        if "warning_margin_multiplier" in safety:
            self.warning_margin_multiplier = float(safety["warning_margin_multiplier"])
        if "critical_margin_multiplier" in safety:
            self.critical_margin_multiplier = float(safety["critical_margin_multiplier"])

        corridor = cfg.get("corridor_params", {})
        if "lane_width_m" in corridor:
            self.lane_width_m = float(corridor["lane_width_m"])
        if "lateral_corridor_half_width_m" in corridor:
            self.lateral_corridor_half_width_m = float(corridor["lateral_corridor_half_width_m"])

        vp = cfg.get("vehicle_priors", {})
        for name, params in vp.items():
            self.priors[name.lower()] = VehiclePrior(
                height_m=float(params.get("height_m", 1.5)),
                width_m=float(params.get("width_m", 1.8)),
                length_m=float(params.get("length_m", 4.5)),
                alpha=float(params.get("alpha", 1.0)),
                max_decel_ms2=float(params.get("max_decel_ms2", 6.0)),
            )

    def get_prior(self, class_name: str) -> VehiclePrior:
        """Fetch prior for class, with fallback to default."""
        return self.priors.get(class_name.lower(), self.priors["default"])

    def estimate_distance(
        self,
        bbox: List[float] | Tuple[float, float, float, float],
        class_name: str,
        image_width: Optional[int] = None,
    ) -> float:
        """
        Estimate 3D longitudinal distance Z (meters) using pinhole camera geometry:
            Z = (f_y * H_prior) / h_bbox
        """
        x1, y1, x2, y2 = bbox
        h_px = max(1.0, float(y2 - y1))

        prior = self.get_prior(class_name)

        if image_width is not None and image_width > 0:
            f_y = (image_width / 2.0) / math.tan(math.radians(self.fov_horizontal_deg / 2.0))
        else:
            f_y = self.focal_length_px

        est_z = (f_y * prior.height_m) / h_px
        return max(0.5, min(200.0, float(est_z)))

    def categorize_size(
        self,
        class_name: str,
        bbox: List[float] | Tuple[float, float, float, float],
    ) -> Tuple[str, float]:
        """
        Determine size category and dimension scaling factor beta_size based on
        object type and bounding box dimensions.
        """
        prior = self.get_prior(class_name)
        x1, y1, x2, y2 = bbox
        w_px = max(1.0, float(x2 - x1))
        h_px = max(1.0, float(y2 - y1))
        aspect_ratio = w_px / h_px

        volume_m3 = prior.length_m * prior.width_m * prior.height_m

        if volume_m3 >= 60.0 or prior.length_m >= 9.0:
            category = "EXTRA_LARGE"  # Truck, Bus
            size_factor = 1.25
        elif volume_m3 >= 10.0 or prior.length_m >= 4.0:
            category = "LARGE"        # Car, Van, SUV
            size_factor = 1.00
        elif volume_m3 >= 1.5:
            category = "MEDIUM"       # Motorcycle
            size_factor = 0.90
        else:
            category = "SMALL"        # Pedestrian, Bicycle
            size_factor = 0.85

        if aspect_ratio > 2.2 and category in ("LARGE", "EXTRA_LARGE"):
            size_factor += 0.15

        return category, size_factor

    def compute_safe_distance_thw(
        self,
        ego_speed_ms: float,
        class_name: str,
        size_factor: float = 1.0,
    ) -> float:
        """
        Compute dynamic safe distance using Time Headway Model:
            d_safe = d_standstill + v_ego * t_hw * alpha_type * beta_size
        """
        v_ego = max(0.0, float(ego_speed_ms))
        prior = self.get_prior(class_name)
        d_safe = self.standstill_distance_m + (
            v_ego * self.time_headway_s * prior.alpha * size_factor
        )
        return float(d_safe)

    def compute_safe_distance_rss(
        self,
        ego_speed_ms: float,
        class_name: str,
        target_speed_ms: Optional[float] = None,
        size_factor: float = 1.0,
    ) -> float:
        """
        Compute safe distance using Kinematic RSS (Responsibility-Sensitive Safety):
            d_safe = max(d_standstill, v_ego*tau + (v_ego^2)/(2*a_ego) - (v_target^2)/(2*a_target) + buffer)
        """
        v_ego = max(0.0, float(ego_speed_ms))
        prior = self.get_prior(class_name)

        v_target = max(0.0, float(target_speed_ms if target_speed_ms is not None else 0.0))

        reaction_dist = v_ego * self.reaction_time_s
        ego_braking_dist = (v_ego ** 2) / (2.0 * self.ego_comfortable_decel_ms2)
        target_braking_dist = (v_target ** 2) / (2.0 * max(1.0, prior.max_decel_ms2))

        size_buffer = (prior.length_m * 0.2) * size_factor if v_ego > 0.0 else 0.0

        d_rss = (
            reaction_dist
            + ego_braking_dist
            - target_braking_dist
            + self.standstill_distance_m
            + size_buffer
        )
        return max(self.standstill_distance_m, float(d_rss))

    def compute_safe_distance(
        self,
        ego_speed_ms: float,
        class_name: str,
        size_factor: float = 1.0,
        target_speed_ms: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute combined safe distance.
        Returns: (final_safe_distance, thw_distance, rss_distance)
        """
        d_thw = self.compute_safe_distance_thw(ego_speed_ms, class_name, size_factor)
        d_rss = self.compute_safe_distance_rss(ego_speed_ms, class_name, target_speed_ms, size_factor)
        final_d = max(d_thw, d_rss)
        return final_d, d_thw, d_rss

    def evaluate_safety(
        self,
        bbox: List[float] | Tuple[float, float, float, float],
        class_name: str,
        ego_speed_ms: float,
        target_speed_ms: Optional[float] = None,
        direct_distance_m: Optional[float] = None,
        image_width: Optional[int] = None,
    ) -> SafetyEvaluation:
        """
        Evaluate full safety profile for a single detection:
        - Bbox dimension breakdown
        - Size categorization
        - 3D longitudinal (Z) and lateral (X) position estimation
        - Lateral lane corridor classification (IN_PATH, RIGHT_LANE, LEFT_LANE, ROADSIDE)
        - Dynamic safe distance calculation
        - Distance margin & unbiased safety status ('SAFE', 'WARNING', 'CRITICAL')
        - Time-To-Collision (TTC) if relative closing speed is available
        """
        x1, y1, x2, y2 = bbox
        w_px = float(x2 - x1)
        h_px = float(y2 - y1)
        area_px = float(w_px * h_px)

        bbox_dims = {
            "width_px": round(w_px, 1),
            "height_px": round(h_px, 1),
            "area_px": round(area_px, 1),
        }

        # Size classification
        size_cat, size_factor = self.categorize_size(class_name, bbox)

        # Distance estimation: use direct measurement if provided, else monocular estimation
        if direct_distance_m is not None and direct_distance_m > 0:
            est_dist = float(direct_distance_m)
        else:
            est_dist = self.estimate_distance(bbox, class_name, image_width=image_width)

        # Compute lateral 3D position (X in meters from vehicle center)
        w_img = image_width if image_width is not None and image_width > 0 else self.image_width
        f_x = (w_img / 2.0) / math.tan(math.radians(self.fov_horizontal_deg / 2.0))
        c_x = w_img / 2.0

        cx_bbox = (x1 + x2) / 2.0
        lateral_offset_m = ((cx_bbox - c_x) * est_dist) / f_x

        # Lane corridor classification
        corridor = self.lateral_corridor_half_width_m
        lane_w = self.lane_width_m

        if abs(lateral_offset_m) <= corridor:
            lane_pos = "IN_PATH"
            in_path = True
        elif lateral_offset_m > corridor and lateral_offset_m <= corridor + lane_w:
            lane_pos = "RIGHT_LANE"
            in_path = False
        elif lateral_offset_m > corridor + lane_w:
            lane_pos = "RIGHT_ROADSIDE"
            in_path = False
        elif lateral_offset_m < -corridor and lateral_offset_m >= -(corridor + lane_w):
            lane_pos = "LEFT_LANE"
            in_path = False
        else:
            lane_pos = "LEFT_ROADSIDE"
            in_path = False

        # Compute safe distances
        final_safe, d_thw, d_rss = self.compute_safe_distance(
            ego_speed_ms=ego_speed_ms,
            class_name=class_name,
            size_factor=size_factor,
            target_speed_ms=target_speed_ms,
        )

        margin = est_dist - final_safe

        # Unbiased Safety Status Classification:
        # 1. Non-vehicle signals (traffic light, stop sign) are informational infrastructure
        if class_name.lower() in ("traffic light", "stop sign"):
            status = "INFO (SIGNAL)"
        # 2. Objects outside our lane corridor are in adjacent lanes or roadside -> SAFE
        elif not in_path:
            status = f"SAFE ({lane_pos})"
        # 3. In-path obstacles pose longitudinal collision risk based on safe distance margin
        else:
            if est_dist < final_safe * self.critical_margin_multiplier:
                status = "CRITICAL"
            elif est_dist < final_safe * self.warning_margin_multiplier:
                status = "WARNING"
            else:
                status = "SAFE"

        # Time-To-Collision (TTC)
        ttc = None
        if target_speed_ms is not None:
            rel_speed = ego_speed_ms - target_speed_ms
            if rel_speed > 0.1:
                ttc = est_dist / rel_speed

        return SafetyEvaluation(
            bbox_dimensions=bbox_dims,
            size_category=size_cat,
            size_factor=size_factor,
            estimated_distance_m=est_dist,
            safe_distance_m=final_safe,
            safe_distance_thw_m=d_thw,
            safe_distance_rss_m=d_rss,
            distance_margin_m=margin,
            safety_status=status,
            lateral_offset_m=lateral_offset_m,
            lane_position=lane_pos,
            in_ego_path=in_path,
            time_to_collision_s=ttc,
        )

    def enrich_detection(
        self,
        detection: Dict[str, Any],
        ego_speed_ms: float = 0.0,
        target_speed_ms: Optional[float] = None,
        direct_distance_m: Optional[float] = None,
        image_width: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Enrich a detection dictionary with safe distance, lateral position, and safety status.
        """
        bbox = detection["bbox"]
        class_name = detection.get("class_name", "car")

        safety_eval = self.evaluate_safety(
            bbox=bbox,
            class_name=class_name,
            ego_speed_ms=ego_speed_ms,
            target_speed_ms=target_speed_ms,
            direct_distance_m=direct_distance_m,
            image_width=image_width,
        )

        enriched = dict(detection)
        enriched.update(safety_eval.to_dict())
        return enriched

    def enrich_detections(
        self,
        detections: List[Dict[str, Any]],
        ego_speed_ms: float = 0.0,
        image_width: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Enrich a batch of detection dictionaries.
        """
        return [
            self.enrich_detection(d, ego_speed_ms=ego_speed_ms, image_width=image_width)
            for d in detections
        ]
