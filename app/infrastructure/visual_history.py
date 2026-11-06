"""Compara produtos por posição física e aparência no histórico da câmera fixa.

Uma detecção isolada não prova ruptura: caixas ausentes exigem mudança visual
real na mesma posição. O modelo YOLO reconhece "object"; os IDs de padrões
são identificadores visuais, não nomes comerciais/SKUs.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / max(1, area_a + area_b - intersection)


def _objects(image, detections):
    height, width = image.shape[:2]
    objects = []
    for d in sorted(detections, key=lambda x: x.confidence, reverse=True):
        b = d.bbox
        box = [max(0, int(b.x1)), max(0, int(b.y1)),
               min(width, int(b.x2)), min(height, int(b.y2))]
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        # Deduplicação apenas de caixas muito sobrepostas; produtos lado a lado
        # não devem ser fundidos.
        if any(_iou(box, other["box"]) >= .65 for other in objects):
            continue
        crop = image[box[1]:box[3], box[0]:box[2]]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16],
                            [0, 180, 0, 256]).flatten()
        hist /= np.linalg.norm(hist) + 1e-8
        objects.append({"box": box, "x": (box[0]+box[2])/(2*width),
                        "y": (box[1]+box[3])/(2*height),
                        "class": d.class_name, "feature": hist})
    return objects


def _shelves(reference, current):
    """Agrupa por linhas visíveis de produtos, sem supor número de prateleiras."""
    if not reference:
        return
    ys = sorted(obj["y"] for obj in reference)
    groups = [[ys[0]]]
    for y in ys[1:]:
        if y - float(np.median(groups[-1])) <= .055:
            groups[-1].append(y)
        else:
            groups.append([y])
    centers = [float(np.median(group)) for group in groups]
    boundaries = [(a + b)/2 for a, b in zip(centers, centers[1:])]
    for item in reference + current:
        item["shelf"] = 1 + int(np.searchsorted(boundaries, item["y"]))


def _match(reference, current):
    """Correspondência espacial única: nenhum objeto atual serve para dois."""
    candidates = []
    for ri, ref in enumerate(reference):
        rb = ref["box"]
        rw = rb[2] - rb[0]
        rh = rb[3] - rb[1]
        for ci, obj in enumerate(current):
            if ref["class"] != obj["class"] or ref["shelf"] != obj["shelf"]:
                continue
            overlap = _iou(rb, obj["box"])
            dx = abs(ref["x"] - obj["x"])
            dy = abs(ref["y"] - obj["y"])
            # A referência da câmera é fixa, logo o mesmo item deve ficar
            # na mesma posição. Tolerância pequena a deslocamentos da caixa.
            if overlap < .18 and (dx > .025 or dy > .035):
                continue
            resemblance = float(ref["feature"] @ obj["feature"])
            score = 2 * overlap + .35 * resemblance - 2 * dx - 2 * dy
            candidates.append((score, ri, ci))
    matches = {}
    assigned_current = set()
    for score, ri, ci in sorted(candidates, reverse=True):
        if ri not in matches and ci not in assigned_current:
            matches[ri] = ci
            assigned_current.add(ci)
    return matches, assigned_current


def _patch_difference(reference, current, box):
    """Mede alteração real na região que antes continha o produto."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = max(1, int(w * .12)), max(1, int(h * .12))
    x1, x2 = x1 + margin_x, x2 - margin_x
    y1, y2 = y1 + margin_y, y2 - margin_y
    if x2 <= x1 or y2 <= y1:
        return 0.0
    a = cv2.GaussianBlur(reference[y1:y2, x1:x2], (5, 5), 0)
    b = cv2.GaussianBlur(current[y1:y2, x1:x2], (5, 5), 0)
    # Mediana reduz a influência de ruído JPEG ou reflexos em poucos pixels.
    return float(np.median(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _patterns(reference):
    """Identidades agrupadas por aparência, dentro de cada prateleira."""
    prototypes = []
    groups = []
    for item in reference:
        compatible = [(float(item["feature"] @ proto["feature"]), j)
                      for j, proto in enumerate(prototypes)
                      if item["shelf"] == proto["shelf"] and
                      item["class"] == proto["class"]]
        if compatible and max(compatible)[0] >= .92:
            index = max(compatible)[1]
            groups.append(index)
            proto = prototypes[index]
            proto["members"].append(item["feature"])
            vector = np.mean(proto["members"], axis=0)
            proto["feature"] = vector / (np.linalg.norm(vector) + 1e-8)
        else:
            groups.append(len(prototypes))
            prototypes.append({"shelf": item["shelf"], "class": item["class"],
                               "feature": item["feature"].copy(),
                               "members": [item["feature"]]})
    return groups



def _align_to_reference(reference, current):
    """Registra a captura sobre a referência usando movimento translacional.
    
    Retorna None quando a câmera mudou demais ou o registro falhou.
    """
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    # Redução de escala torna o ECC mais estável para JPEG da ESP32-CAM.
    size = (max(80, reference.shape[1] // 2), max(60, reference.shape[0] // 2))
    a = cv2.resize(ref_gray, size).astype(np.float32) / 255.0
    b = cv2.resize(cur_gray, size).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        corr, warp = cv2.findTransformECC(
            a, b, warp, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
            None, 3)
    except cv2.error:
        return None
    warp[0, 2] *= reference.shape[1] / size[0]
    warp[1, 2] *= reference.shape[0] / size[1]
    shift_x, shift_y = float(warp[0, 2]), float(warp[1, 2])
    if not np.isfinite(corr) or corr < .60 or abs(shift_x) > 25 or abs(shift_y) > 25:
        return None
    aligned = cv2.warpAffine(
        current, warp, (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE)
    return aligned, round(corr, 3), [round(shift_x, 2), round(shift_y, 2)]


class VisualHistory:
    def __init__(self, directory: Path, detector):
        self.directory = directory
        self.detector = detector
        self.lock = threading.RLock()
        directory.mkdir(parents=True, exist_ok=True)

    def _path(self, camera_id):
        return self.directory / f"{camera_id}.jpg"

    def save_reference(self, camera_id: str, data: bytes) -> dict:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Imagem histórica inválida")
        if not self.detector.is_available:
            raise ValueError("Modelo de produtos indisponível")
        detections = self.detector.detect(image)
        objects = _objects(image, detections)
        if not objects:
            raise ValueError("A referência não contém produtos detectados")
        with self.lock:
            # Mantém os bytes originais JPEG para comparações repetíveis.
            # PNG também é aceito e convertido para JPEG.
            if data[:2] == bytes((255, 216)):
                self._path(camera_id).write_bytes(data)
            else:
                ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 98])
                if not ok:
                    raise ValueError("Falha ao codificar referência")
                self._path(camera_id).write_bytes(jpg.tobytes())
        return {"camera_id": camera_id, "referencia_salva": True,
                "objetos_referencia": len(objects)}

    def analyze(self, camera_id: str, current_image: np.ndarray, current_detections) -> dict:
        path = self._path(camera_id)
        with self.lock:
            if not path.is_file():
                return {"status": "sem_referencia", "tem_ruptura": None, "rupturas": [],
                        "motivo": "Cadastre uma imagem histórica em POST /reference"}
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
        if image is None:
            return {"status": "referencia_invalida", "tem_ruptura": None, "rupturas": []}
        if image.shape[:2] != current_image.shape[:2]:
            return {"status": "resolucao_diferente", "tem_ruptura": None, "rupturas": [],
                    "motivo": "Referência e captura devem ter a mesma resolução"}

        # Normalização geométrica: comparar a mesma posição antes/depois.
        # Reexecutamos o detector sobre a imagem alinhada para não misturar
        # caixas originais com coordenadas do histórico.
        registration = _align_to_reference(image, current_image)
        if registration is None:
            return {"status": "indeterminado", "tem_ruptura": None, "rupturas": [],
                    "motivo": "Não foi possível alinhar a câmera com o histórico"}
        aligned, registration_score, deslocamento = registration

        global_diff = float(np.median(np.abs(
            image.astype(np.float32) - aligned.astype(np.float32))))
        ref = _objects(image, self.detector.detect(image))
        if not ref:
            return {"status": "referencia_sem_deteccoes", "tem_ruptura": None, "rupturas": []}
        if global_diff < 2.5:
            return {"status": "ok", "tem_ruptura": False, "rupturas": [],
                    "objetos_referencia": len(ref), "objetos_atuais": len(ref),
                    "objetos_sem_tipo": 0, "deslocamento_pixels": deslocamento,
                    "observacao": "Cena equivalente ao histórico"}
        cur = _objects(aligned, self.detector.detect(aligned))

        _shelves(ref, cur)
        matches, used = _match(ref, cur)
        patterns = _patterns(ref)
        missing = {}
        uncertain = 0
        for i, obj in enumerate(ref):
            if i in matches:
                continue
            change = _patch_difference(image, aligned, obj["box"])
            # Sem mudança real no local, a falta de detecção não é ruptura.
            if change < 12:
                continue
            # Não confirmar como ruptura se um outro objeto atual cobre a
            # região: pode ter mudado a embalagem, posição ou classificação.
            occupied = any(_iou(obj["box"], other["box"]) >= .20 for other in cur)
            if occupied:
                uncertain += 1
                continue
            pattern_id = patterns[i]
            missing.setdefault(pattern_id, []).append(
                {"box": obj["box"], "alteracao_visual": round(change, 2)})
        counts = {}
        for i, pattern_id in enumerate(patterns):
            counts[pattern_id] = counts.get(pattern_id, 0) + 1
        ruptures = []
        for pattern_id, positions in sorted(missing.items()):
            ruptures.append({
                "produto": f"padrao_{pattern_id+1:03d}",
                "prateleira": next(ref[i]["shelf"] for i in range(len(ref))
                                   if patterns[i] == pattern_id),
                "esperado": counts[pattern_id],
                "encontrado": max(0, counts[pattern_id] - len(positions)),
                "faltando": len(positions),
                "posicoes_historicas": [position["box"] for position in positions],
            })
        # A ausência de detecção com mudança de cena não confirma remoção.
        # Expor uma decisão positiva somente para regiões claras, sem objetos.
        status = "indeterminado" if uncertain else "ok"
        return {"status": status,
                "tem_ruptura": None if uncertain else bool(ruptures),
                "rupturas": ruptures, "objetos_referencia": len(ref),
                "objetos_atuais": len(cur),
                "objetos_sem_tipo": len(cur) - len(used),
                "regioes_incertas": uncertain,
                "confianca_alinhamento": registration_score,
                "deslocamento_pixels": deslocamento,
                "observacao": "Rupturas estimadas por comparação espacial e visual. "
                              "Atribuição é por padrão visual, não por SKU."}
