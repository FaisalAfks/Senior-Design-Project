from .device import resolve_device
from .camera import (
    CameraConfig,
    DEFAULT_CAMERA_CONFIG,
    build_nvargus_pipeline,
    detect_jetson,
    open_capture,
)
from .cv import draw_detection_labels
from .face_guidance import (
    AlignmentAssessment,
    draw_guidance_overlay,
    evaluate_alignment,
    run_guidance_session,
    select_best_detection,
)
from .verification import (
    FaceObservation,
    aggregate_observations,
    append_attendance_log,
    compose_final_display,
    evaluate_frame,
)

__all__ = [
    'resolve_device',
    'draw_detection_labels',
    'AlignmentAssessment',
    'draw_guidance_overlay',
    'evaluate_alignment',
    'run_guidance_session',
    'select_best_detection',
    'FaceObservation',
    'evaluate_frame',
    'aggregate_observations',
    'append_attendance_log',
    'compose_final_display',
    'open_capture',
    'build_nvargus_pipeline',
    'detect_jetson',
    'CameraConfig',
    'DEFAULT_CAMERA_CONFIG',
]

