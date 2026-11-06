"""Comparação de ruptura por padrão visual entre referência e captura atual.

Não atribui nomes comerciais: padrao_N é a identidade visual inferida.
Referência deve ser da mesma câmera fixa e gôndola abastecida.
"""
from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


class VisualHistory:
    def __init__(self, directory: Path, detector):
        self.directory = directory
        self.detector = detector
        self.lock = threading.RLock()
        directory.mkdir(parents=True, exist_ok=True)

    def _path(self, camera_id: str) -> Path:
        return self.directory / f"{camera_id}.jpg"

    def save_reference(self, camera_id: str, data: bytes) -> dict:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Imagem histórica inválida")
        if not self.detector.is_available:
            raise ValueError("Modelo de produtos indisponível")
        detections = self.detector.detect(image)
        if not detections:
            raise ValueError("A referência não contém objetos detectados")
        with self.lock:
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 96])
            if not ok:
                raise ValueError("Falha ao codificar referência")
            self._path(camera_id).write_bytes(buf.tobytes())
        return {"camera_id": camera_id, "referencia_salva": True,
                "objetos_referencia": len(detections)}

    @staticmethod
    def _features(image: np.ndarray, detections) -> list[dict]:
        h, w = image.shape[:2]
        items = []
        for d in detections:
            b = d.bbox
            x1, y1, x2, y2 = max(0, int(b.x1)), max(0, int(b.y1)), min(w, int(b.x2)), min(h, int(b.y2))
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            crop = image[y1:y2, x1:x2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
            hist /= np.linalg.norm(hist) + 1e-8
            lab = cv2.cvtColor(cv2.resize(crop, (8, 8)), cv2.COLOR_BGR2LAB)
            shape = lab[:, :, 1:].astype(np.float32).flatten() / 255.0
            shape -= shape.mean()
            shape /= np.linalg.norm(shape) + 1e-8
            feature = np.concatenate((hist * .8, shape * .2))
            feature /= np.linalg.norm(feature) + 1e-8
            items.append({"box": [x1, y1, x2, y2], "x": (x1+x2)/2 / w,
                          "y": (y1+y2)/2 / h, "f": feature, "class": d.class_name})
        return items

    @staticmethod
    def _shelf_groups(ref: list[dict], cur: list[dict]) -> None:
        if not ref:
            return
        centers = sorted(r["y"] for r in ref)
        heights = [r["box"][3] - r["box"][1] for r in ref]
        # Normalização da altura é efetuada a partir da própria imagem.
        image_h = max(1, max(r["box"][3] for r in ref))
        tolerance = max(.025, float(np.median(heights)) / image_h * .65)
        rows = [[centers[0]]]
        for y in centers[1:]:
            if y - float(np.mean(rows[-1])) < tolerance:
                rows[-1].append(y)
            else:
                rows.append([y])
        marks = [float(np.mean(v)) for v in rows]
        cuts = [(a+b)/2 for a, b in zip(marks, marks[1:])]
        for item in ref + cur:
            item["shelf"] = 1 + int(np.searchsorted(cuts, item["y"]))

    def analyze(self, camera_id: str, current_image: np.ndarray, current_detections) -> dict:
        path = self._path(camera_id)
        with self.lock:
            if not path.is_file():
                return {"status": "sem_referencia", "tem_ruptura": None, "rupturas": [],
                        "motivo": "Cadastre uma imagem histórica em POST /reference"}
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"status": "referencia_invalida", "tem_ruptura": None, "rupturas": []}
        if image.shape[:2] != current_image.shape[:2]:
            return {"status": "resolucao_diferente", "tem_ruptura": None, "rupturas": [],
                    "motivo": "Referência e captura precisam ter a mesma resolução"}
        ref = self._features(image, self.detector.detect(image))
        cur = self._features(current_image, current_detections)
        if not ref:
            return {"status": "referencia_sem_deteccoes", "tem_ruptura": None, "rupturas": []}
        self._shelf_groups(ref, cur)
        prototypes: list[dict] = []
        # Grupos de produtos por aparência dentro de cada prateleira.
        for item in sorted(ref, key=lambda x: (x["shelf"], x["x"])):
            sims = [(float(item["f"] @ p["feature"]), j)
                    for j, p in enumerate(prototypes)
                    if item["shelf"] == p["shelf"] and item["class"] == p["class"]]
            if sims and max(sims)[0] >= .88:
                p = prototypes[max(sims)[1]]
                p["items"].append(item)
                vec = np.mean([x["f"] for x in p["items"]], axis=0)
                p["feature"] = vec / (np.linalg.norm(vec) + 1e-8)
            else:
                prototypes.append({"shelf": item["shelf"], "class": item["class"],
                                   "feature": item["f"].copy(), "items": [item]})
        observed = Counter()
        unknown = 0
        for item in cur:
            sims = [(float(item["f"] @ p["feature"]), j) for j, p in enumerate(prototypes)
                    if item["shelf"] == p["shelf"] and item["class"] == p["class"]]
            if sims and max(sims)[0] >= .77:
                observed[max(sims)[1]] += 1
            else:
                unknown += 1
        ruptures = []
        for j, p in enumerate(prototypes):
            expected, found = len(p["items"]), observed[j]
            if found >= expected:
                continue
            ruptures.append({"produto": f"padrao_{j+1:03d}", "prateleira": p["shelf"],
                             "esperado": expected, "encontrado": found, "faltando": expected-found,
                             "posicoes_historicas": [r["box"] for r in p["items"]]})
        # Objetos atuais não associados tornam o resultado incerto para evitar falso positivo.
        status = "indeterminado" if unknown else "ok"
        return {"status": status, "tem_ruptura": bool(ruptures) if not unknown else None,
                "rupturas": ruptures, "objetos_referencia": len(ref),
                "objetos_atuais": len(cur), "objetos_sem_tipo": unknown,
                "observacao": "Comparação visual aproximada; caixas históricas não são coordenadas de falta confirmada."}
