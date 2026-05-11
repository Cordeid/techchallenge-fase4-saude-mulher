"""
Nó VIDEO do pipeline: analisa emoções faciais (DeepFace/CPU), postura corporal
(MediaPipe PoseLandmarker Tasks API) e objetos em cena (YOLOv8/CUDA).

Compatível com MediaPipe >= 0.10 (Tasks API apenas — solutions API removida).
TensorFlow é forçado a usar CPU via tf.config (sem afetar PyTorch/YOLO).
"""

import logging
import os
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

from src.config import CACHE_DIR, INTERMEDIATE_DIR, configurar_logging
from src.schemas import (
    FacialEmotionEvent,
    ObjectDetectionEvent,
    PostureEvent,
    VideoAnalysis,
)

logger = logging.getLogger(__name__)

# Classes COCO relevantes para contexto clínico
CLINICAL_CLASSES = {
    "person", "knife", "scissors", "bottle",
    "cell phone", "book", "laptop", "chair",
}

# Conexões do esqueleto MediaPipe Pose 33 landmarks (para desenho manual)
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
])

# Índices dos landmarks relevantes para postura
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW,    _R_ELBOW    = 13, 14
_L_WRIST,    _R_WRIST    = 15, 16
_L_HIP,      _R_HIP      = 23, 24

# URL do modelo PoseLandmarker lite (~5 MB) — Google MediaPipe CDN
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

_TF_CPU_CONFIGURED = False


# ── Configuração de GPU ───────────────────────────────────────────────────────

def _configurar_tf_cpu() -> None:
    """Força TensorFlow a usar CPU sem afetar PyTorch (YOLO/Whisper)."""
    global _TF_CPU_CONFIGURED
    if _TF_CPU_CONFIGURED:
        return
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        logger.debug("TensorFlow configurado para CPU via tf.config")
    except Exception as exc:
        # Fallback conservador — afeta PyTorch também, mas é seguro
        logger.debug("Fallback CUDA_VISIBLE_DEVICES=-1 (tf.config indisponível): %s", exc)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    _TF_CPU_CONFIGURED = True


def _cuda_disponivel() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── MediaPipe PoseLandmarker (Tasks API) ──────────────────────────────────────

def _baixar_modelo_pose() -> Path:
    """Retorna path do modelo PoseLandmarker, baixando se necessário."""
    model_path = CACHE_DIR / "mediapipe" / "pose_landmarker_lite.task"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        logger.info("Baixando modelo MediaPipe PoseLandmarker lite (~5MB)...")
        urllib.request.urlretrieve(_POSE_MODEL_URL, model_path)
        logger.info("Modelo salvo em: %s", model_path)
    return model_path


def _criar_landmarker(model_path: Path) -> mp_vision.PoseLandmarker:
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


# ── Helpers de análise ────────────────────────────────────────────────────────

def _calcular_angulo(p1: tuple, p2: tuple, p3: tuple) -> float:
    """Ângulo em graus no ponto p2 formado por p1-p2-p3."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def _classificar_postura(landmarks: list) -> tuple[str, dict]:
    """Classifica postura corporal a partir de landmarks normalizados [0,1].

    Estratégia:
      defensiva → pulso esquerdo cruza para a direita do centro e vice-versa
      encolhida → ombros estreitos (<0.15) OU cotovelos dobrados (<110°) + pulsos no tronco
      aberta    → pulsos afastados do centro (>0.20 de cada lado)
      neutra    → padrão
    """
    ls = landmarks[_L_SHOULDER]
    rs = landmarks[_R_SHOULDER]
    le = landmarks[_L_ELBOW]
    re = landmarks[_R_ELBOW]
    lw = landmarks[_L_WRIST]
    rw = landmarks[_R_WRIST]
    lh = landmarks[_L_HIP]
    rh = landmarks[_R_HIP]

    midline_x     = (ls.x + rs.x) / 2
    shoulder_width = abs(ls.x - rs.x)
    shoulder_mid_y = (ls.y + rs.y) / 2
    hip_mid_y      = (lh.y + rh.y) / 2

    ang_l = _calcular_angulo((ls.x, ls.y), (le.x, le.y), (lw.x, lw.y))
    ang_r = _calcular_angulo((rs.x, rs.y), (re.x, re.y), (rw.x, rw.y))

    summary = {
        "shoulder_width":        round(shoulder_width, 3),
        "midline_x":             round(midline_x, 3),
        "left_wrist_x":          round(lw.x, 3),
        "right_wrist_x":         round(rw.x, 3),
        "elbow_angle_left_deg":  round(ang_l, 1),
        "elbow_angle_right_deg": round(ang_r, 1),
    }

    arms_crossed   = (lw.x > midline_x + 0.02) and (rw.x < midline_x - 0.02)
    ombros_str     = shoulder_width < 0.15
    bracos_dob     = (ang_l < 110.0) and (ang_r < 110.0)
    pulsos_tronco  = (
        shoulder_mid_y < lw.y < hip_mid_y
        and shoulder_mid_y < rw.y < hip_mid_y
    )
    encolhida = ombros_str or (bracos_dob and pulsos_tronco)
    aberta    = (lw.x < midline_x - 0.20) and (rw.x > midline_x + 0.20)

    if arms_crossed:
        return "defensiva", summary
    if encolhida:
        return "encolhida", summary
    if aberta:
        return "aberta", summary
    return "neutra", summary


def _analisar_emocao(frame_bgr: np.ndarray) -> Optional[tuple[str, float, dict, dict]]:
    """Executa DeepFace (CPU) e retorna (dominant, confidence, all_emotions, region).

    Retorna None se nenhuma face for detectada.
    """
    _configurar_tf_cpu()
    from deepface import DeepFace  # import tardio — TF já está configurado para CPU

    try:
        resultados = DeepFace.analyze(
            img_path=frame_bgr,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
            detector_backend="opencv",
        )
        if not resultados:
            return None
        r = resultados[0]
        dominant    = r["dominant_emotion"]
        all_em      = {k: round(v / 100.0, 4) for k, v in r["emotion"].items()}
        confidence  = all_em.get(dominant, 0.0)
        region      = r.get("region", {})
        return dominant, confidence, all_em, region
    except Exception as exc:
        logger.debug("DeepFace não detectou face: %s", exc)
        return None


def _detectar_objetos(frame_bgr: np.ndarray, model, device: str) -> list[dict]:
    """YOLOv8: retorna apenas objetos das classes clínicas."""
    resultados = model(frame_bgr, device=device, verbose=False)
    objetos = []
    for r in resultados:
        for box in r.boxes:
            nome = model.names[int(box.cls)]
            if nome not in CLINICAL_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            objetos.append({
                "class":      nome,
                "confidence": round(float(box.conf), 3),
                "bbox":       [round(x1), round(y1), round(x2), round(y2)],
            })
    return objetos


# ── Desenho de anotações ──────────────────────────────────────────────────────

def _desenhar_esqueleto(frame: np.ndarray, landmarks: list, h: int, w: int) -> None:
    """Desenha skeleton MediaPipe manualmente com cv2 (Tasks API não tem drawing_utils)."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for i, (cx, cy) in enumerate(pts):
        vis = getattr(landmarks[i], "visibility", 1.0)
        if vis < 0.3:
            continue
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            va = getattr(landmarks[a], "visibility", 1.0)
            vb = getattr(landmarks[b], "visibility", 1.0)
            if va < 0.3 or vb < 0.3:
                continue
            cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2)


def _desenhar_anotacoes(
    frame: np.ndarray,
    face_region:      Optional[dict],
    dominant_emotion: str,
    posture_label:    str,
    yolo_objects:     list[dict],
    timestamp_s:      float,
    pose_landmarks:   Optional[list],
) -> None:
    """Sobrepõe todas as anotações no frame (in-place)."""
    h, w = frame.shape[:2]

    # Esqueleto de pose
    if pose_landmarks:
        _desenhar_esqueleto(frame, pose_landmarks, h, w)

    # Retângulo + label da face (DeepFace region em pixels)
    if face_region and face_region.get("w", 0) > 0:
        x  = face_region.get("x", 0)
        y  = face_region.get("y", 0)
        fw = face_region.get("w", 0)
        fh = face_region.get("h", 0)
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        cv2.putText(
            frame, dominant_emotion,
            (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    # Bounding boxes YOLO
    for obj in yolo_objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = f"{obj['class']} {obj['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(
            frame, label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2,
        )

    # HUD inferior
    hud = f"tempo: {timestamp_s:.1f}s | emocao: {dominant_emotion} | postura: {posture_label}"
    cv2.rectangle(frame, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, hud,
        (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )


# ── Função principal ──────────────────────────────────────────────────────────

def analyze_video(
    video_path:            Path,
    output_video_path:     Path,
    sample_every_n_frames: int = 15,
) -> VideoAnalysis:
    """Analisa vídeo clínico e retorna métricas multimodais + vídeo anotado.

    Processa emoções faciais (DeepFace/CPU), postura (MediaPipe PoseLandmarker)
    e objetos (YOLOv8) em frames amostrados. Todos os frames recebem anotações.

    Args:
        video_path: Caminho para o vídeo de entrada.
        output_video_path: Caminho para o vídeo anotado de saída.
        sample_every_n_frames: Intervalo de amostragem (padrão 15).

    Returns:
        VideoAnalysis com eventos por frame e estatísticas agregadas.

    Raises:
        FileNotFoundError: Se o vídeo não existir.
        RuntimeError: Se o vídeo não puder ser aberto.
    """
    log_path = INTERMEDIATE_DIR / "video_node.log"
    configurar_logging(arquivo_log=log_path)
    logger.info("Iniciando análise de vídeo: %s", video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    # ── Carregamento dos modelos ──────────────────────────────────────────────
    logger.info("Carregando modelos...")

    device = "cuda" if _cuda_disponivel() else "cpu"
    logger.info("Dispositivo PyTorch/YOLO: %s", device)

    # YOLO (PyTorch — carrega ANTES de configurar TF para CPU)
    from ultralytics import YOLO
    yolo_model_path = CACHE_DIR / "ultralytics" / "yolov8n.pt"
    yolo_model_path.parent.mkdir(parents=True, exist_ok=True)
    if yolo_model_path.exists():
        yolo = YOLO(str(yolo_model_path))
        logger.info("YOLOv8n carregado do cache")
    else:
        logger.info("Baixando YOLOv8n (~6MB)...")
        yolo = YOLO("yolov8n.pt")
        import shutil
        for candidate in [
            Path.home() / ".cache" / "ultralytics" / "assets" / "yolov8n.pt",
            Path("yolov8n.pt"),
        ]:
            if candidate.exists():
                shutil.copy(candidate, yolo_model_path)
                logger.info("YOLOv8n salvo em: %s", yolo_model_path)
                break

    # MediaPipe PoseLandmarker
    pose_model_path = _baixar_modelo_pose()
    landmarker = _criar_landmarker(pose_model_path)
    logger.info("PoseLandmarker carregado")

    # TF/DeepFace — configura CPU depois do YOLO para não afetar PyTorch
    _configurar_tf_cpu()

    logger.info("Modelos prontos. Abrindo vídeo...")

    # ── Abertura do vídeo ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps

    logger.info(
        "Vídeo: %dx%d | %.1ffps | %d frames | %.1fs",
        width, height, fps, total_frames, duration_s,
    )

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # ── Estruturas de coleta ──────────────────────────────────────────────────
    emotion_events:  list[FacialEmotionEvent]   = []
    posture_events:  list[PostureEvent]          = []
    object_events:   list[ObjectDetectionEvent]  = []
    frames_analisados = 0

    # Estado "último conhecido" para anotar frames não amostrados
    last_emotion:        str            = "neutra"
    last_face_region:    Optional[dict] = None
    last_posture:        str            = "neutra"
    last_yolo_objects:   list[dict]     = []
    last_pose_landmarks: Optional[list] = None

    # ── Loop principal ────────────────────────────────────────────────────────
    logger.info(
        "Processando %d frames (amostragem: 1 a cada %d)...",
        total_frames, sample_every_n_frames,
    )

    with tqdm(total=total_frames, desc="Analisando vídeo", unit="frame") as pbar:
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                logger.warning("Leitura interrompida no frame %d", frame_idx)
                break

            timestamp_s  = frame_idx / fps
            timestamp_ms = int(frame_idx * 1000 / fps)

            if frame_idx % sample_every_n_frames == 0:
                frames_analisados += 1

                # DeepFace (CPU)
                em_result = _analisar_emocao(frame)
                if em_result is not None:
                    dom, conf, all_em, region = em_result
                    last_emotion     = dom
                    last_face_region = region
                    emotion_events.append(FacialEmotionEvent(
                        timestamp_seconds=timestamp_s,
                        dominant_emotion=dom,
                        confidence=conf,
                        all_emotions=all_em,
                    ))
                    logger.debug("Frame %d | emoção: %s (%.0f%%)", frame_idx, dom, conf * 100)

                # MediaPipe PoseLandmarker (Tasks API, VIDEO mode)
                frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if pose_result.pose_landmarks:
                    lms = pose_result.pose_landmarks[0]
                    last_pose_landmarks = lms
                    label, summary = _classificar_postura(lms)
                    last_posture = label
                    posture_events.append(PostureEvent(
                        timestamp_seconds=timestamp_s,
                        posture_label=label,
                        landmarks_summary=summary,
                    ))
                    logger.debug("Frame %d | postura: %s", frame_idx, label)

                # YOLOv8
                objetos = _detectar_objetos(frame, yolo, device)
                last_yolo_objects = objetos
                object_events.append(ObjectDetectionEvent(
                    timestamp_seconds=timestamp_s,
                    detected_objects=objetos,
                ))
                logger.debug(
                    "Frame %d | objetos: %s", frame_idx,
                    [o["class"] for o in objetos],
                )

            # Anota todos os frames com último estado conhecido
            annotated = frame.copy()
            _desenhar_anotacoes(
                annotated,
                last_face_region,
                last_emotion,
                last_posture,
                last_yolo_objects,
                timestamp_s,
                last_pose_landmarks,
            )
            out.write(annotated)
            pbar.update(1)

    cap.release()
    out.release()
    landmarker.close()
    logger.info("Vídeo anotado salvo: %s", output_video_path)

    # ── Estatísticas agregadas ────────────────────────────────────────────────
    if emotion_events:
        contagem        = Counter(e.dominant_emotion for e in emotion_events)
        dominant_overall = contagem.most_common(1)[0][0]
        total_em        = sum(contagem.values())
        emotion_dist    = {k: round(v / total_em, 4) for k, v in contagem.items()}
    else:
        dominant_overall = "indeterminado"
        emotion_dist     = {}

    posturas_def = {"defensiva", "encolhida"}
    ratio = (
        sum(1 for p in posture_events if p.posture_label in posturas_def) / len(posture_events)
        if posture_events else 0.0
    )

    logger.info(
        "Análise concluída | frames analisados: %d | emoção dominante: %s | postura defensiva: %.1f%%",
        frames_analisados, dominant_overall, ratio * 100,
    )

    return VideoAnalysis(
        duration_seconds=round(duration_s, 2),
        fps=round(fps, 2),
        total_frames_analyzed=frames_analisados,
        emotion_events=emotion_events,
        posture_events=posture_events,
        object_events=object_events,
        dominant_emotion_overall=dominant_overall,
        emotion_distribution=emotion_dist,
        defensive_posture_ratio=round(ratio, 4),
    )
