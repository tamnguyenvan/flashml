from __future__ import annotations

import io
import json
import logging
import math
import struct
import sys
import threading
import zipfile
from typing import Any

from flashml.config import Settings
from flashml.errors import InferenceError, InvalidImageError, PayloadTooLargeError
from flashml.schemas import ServiceStatus
from flashml.services.proxy import InferenceProxy

logger = logging.getLogger(__name__)


def _embed_camera_in_glb(
    glb_bytes: bytes,
    intrinsics,
    image_width: int,
    image_height: int,
) -> bytes:
    if len(glb_bytes) < 12:
        return glb_bytes

    magic, version, total_length = struct.unpack_from("<III", glb_bytes)
    if magic != 0x46546C67 or version != 2 or total_length != len(glb_bytes):
        return glb_bytes

    chunks: list[tuple[int, bytes]] = []
    json_index = None
    position = 12
    while position + 8 <= len(glb_bytes):
        chunk_length, chunk_type = struct.unpack_from("<II", glb_bytes, position)
        chunk_end = position + 8 + chunk_length
        if chunk_end > len(glb_bytes):
            return glb_bytes
        if chunk_type == 0x4E4F534A:
            json_index = len(chunks)
        chunks.append((chunk_type, glb_bytes[position + 8 : chunk_end]))
        position = chunk_end

    if position != len(glb_bytes) or json_index is None:
        return glb_bytes

    try:
        gltf = json.loads(chunks[json_index][1].rstrip(b" \t\r\n\0"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return glb_bytes

    fy = float(intrinsics[1, 1])
    camera_index = len(gltf.setdefault("cameras", []))
    gltf["cameras"].append(
        {
            "name": "input_view",
            "type": "perspective",
            "perspective": {
                "aspectRatio": image_width / image_height,
                "yfov": 2.0 * math.atan(0.5 / fy),
                "znear": 0.001,
                "zfar": 1000.0,
            },
        }
    )

    camera_node_index = len(gltf.setdefault("nodes", []))
    gltf["nodes"].append({"camera": camera_index, "name": "camera_input"})

    scenes = gltf.setdefault("scenes", [{"nodes": []}])
    scene_index = gltf.get("scene", 0)
    if not isinstance(scene_index, int) or not 0 <= scene_index < len(scenes):
        scene_index = 0
        gltf["scene"] = scene_index
    scenes[scene_index].setdefault("nodes", []).append(camera_node_index)

    json_chunk = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    chunks[json_index] = (0x4E4F534A, json_chunk)

    body = b"".join(
        struct.pack("<II", len(chunk), chunk_type) + chunk
        for chunk_type, chunk in chunks
    )
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def _png_bytes(image) -> bytes:
    from PIL import Image

    output = io.BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def _mesh_glb(image, points, normal, mask, intrinsics) -> bytes:
    import numpy as np
    import trimesh
    import trimesh.visual
    from PIL import Image

    try:
        import utils3d_moge as utils3d
    except ImportError:
        import utils3d

    height, width = image.shape[:2]
    image_float = image.astype(np.float32) / 255.0
    uv_map = utils3d.np.uv_map(height, width)

    if normal is None:
        faces, vertices, _, vertex_uvs = utils3d.np.build_mesh_from_map(
            points,
            image_float,
            uv_map,
            mask=mask,
            tri=True,
        )
        vertex_normals = None
    else:
        faces, vertices, _, vertex_uvs, vertex_normals = utils3d.np.build_mesh_from_map(
            points,
            image_float,
            uv_map,
            normal,
            mask=mask,
            tri=True,
        )

    coordinate_flip = np.array([1, -1, -1], dtype=np.float32)
    vertices = vertices * coordinate_flip
    vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32)
    vertex_uvs += np.array([0, 1], dtype=np.float32)
    if vertex_normals is not None:
        vertex_normals = vertex_normals * coordinate_flip

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        visual=trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(image),
                metallicFactor=0.5,
                roughnessFactor=1.0,
            ),
        ),
        process=False,
    )
    return _embed_camera_in_glb(mesh.export(file_type="glb"), intrinsics, width, height)


class MogeService:
    backend = "local"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ready = False
        self._lock = threading.Lock()
        self.model = None

    def preload(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._load_locked()

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            enabled=True,
            backend=self.backend,
            ready=self._ready,
            detail=self.settings.moge_model_repo,
        )

    def reconstruct(
        self,
        raw: bytes,
        *,
        filename: str,
        include_mesh: bool,
        include_debug: bool,
        max_size: int,
        resolution_level: int,
        num_tokens: int | None,
        refine_steps: int,
        fov_x: float | None,
        edge_threshold: float,
    ) -> bytes:
        if len(raw) > self.settings.max_upload_bytes:
            raise PayloadTooLargeError(self.settings.max_upload_bytes)
        self.preload()
        with self._lock:
            return self._infer_locked(
                raw,
                filename=filename,
                include_mesh=include_mesh,
                include_debug=include_debug,
                max_size=max_size,
                resolution_level=resolution_level,
                num_tokens=num_tokens,
                refine_steps=refine_steps,
                fov_x=fov_x,
                edge_threshold=edge_threshold,
            )

    def _load_locked(self) -> None:
        root = str(self.settings.moge_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        import torch
        from moge.model.v3 import MoGeModel

        if self.settings.require_cuda and not torch.cuda.is_available():
            raise InferenceError("CUDA is required for MoGe but is not available")

        logger.info("Loading MoGe-3 (%s)", self.settings.moge_model_repo)
        device = self.settings.device if torch.cuda.is_available() else "cpu"
        self.model = MoGeModel.from_pretrained(self.settings.moge_model_repo).to(device).eval()
        warmup = torch.zeros((3, 64, 64), dtype=torch.float32, device=device)
        self.model.infer(
            warmup,
            apply_mask=True,
            refine_steps=1,
            resolution_level=0,
            use_fp16=device == "cuda",
        )
        if device == "cuda":
            torch.cuda.synchronize()
        self._device = device
        self._ready = True
        logger.info("MoGe-3 ready on %s", device)

    def _infer_locked(self, raw: bytes, **kwargs: Any) -> bytes:
        import cv2
        import numpy as np
        import torch
        from moge.utils.vis import colorize_depth, colorize_normal

        try:
            import utils3d_moge as utils3d
        except ImportError:
            import utils3d

        image_array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("Invalid image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_height, original_width = image.shape[:2]
        max_size = kwargs["max_size"]
        if max(original_height, original_width) > max_size:
            scale = max_size / max(original_height, original_width)
            resized_width = max(1, round(original_width * scale))
            resized_height = max(1, round(original_height * scale))
            image = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )

        height, width = image.shape[:2]
        image_tensor = (
            torch.from_numpy(image.copy())
            .to(device=self._device, dtype=torch.float32)
            .permute(2, 0, 1)
            / 255.0
        )
        try:
            output = self.model.infer(
                image_tensor,
                apply_mask=True,
                fov_x=kwargs["fov_x"],
                num_tokens=kwargs["num_tokens"],
                refine_steps=kwargs["refine_steps"],
                resolution_level=kwargs["resolution_level"],
                use_fp16=self._device == "cuda",
            )
        except Exception as exc:
            logger.exception("MoGe inference failed")
            raise InferenceError("MoGe inference failed") from exc

        points = output["points"].cpu().numpy()
        depth = output["depth"].cpu().numpy()
        mask = output["mask"].cpu().numpy()
        intrinsics = output["intrinsics"].cpu().numpy()
        normal_tensor = output.get("normal")
        normal = normal_tensor.cpu().numpy() if normal_tensor is not None else None

        mask_cleaned = mask & ~utils3d.np.depth_map_edge(
            depth,
            rtol=kwargs["edge_threshold"],
        )

        coordinate_flip = np.array([1, -1, -1], dtype=np.float32)
        point_map = points.astype(np.float32, copy=True) * coordinate_flip
        point_map[~mask_cleaned] = np.nan
        point_map_buffer = io.BytesIO()
        np.save(point_map_buffer, point_map, allow_pickle=False)

        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
            if kwargs["include_mesh"]:
                archive.writestr(
                    "output.glb",
                    _mesh_glb(image, points, normal, mask_cleaned, intrinsics),
                )
            archive.writestr("point_map.npy", point_map_buffer.getvalue())
            if kwargs["include_debug"]:
                depth_visualization = depth.copy()
                depth_visualization[~mask_cleaned] = np.nan
                archive.writestr("depth.png", _png_bytes(colorize_depth(depth_visualization)))
                if normal is not None:
                    normal_visualization = normal.copy()
                    normal_visualization[~mask_cleaned] = 0
                    archive.writestr(
                        "normal.png",
                        _png_bytes(colorize_normal(normal_visualization)),
                    )

            fov_x_deg = float(np.rad2deg(2.0 * np.arctan(0.5 / intrinsics[0, 0])))
            fov_y_deg = float(np.rad2deg(2.0 * np.arctan(0.5 / intrinsics[1, 1])))
            metadata = {
                "model": self.settings.moge_model_repo,
                "source_revision": self.settings.moge_source_revision,
                "original_image_size": [original_width, original_height],
                "image_size": [width, height],
                "intrinsics": intrinsics.tolist(),
                "intrinsics_convention": (
                    "normalized by image width/height (principal_point=0.5,0.5)"
                ),
                "fov_x_deg": fov_x_deg,
                "fov_y_deg": fov_y_deg,
                "point_map_coordinates": (
                    "output.glb camera coordinates (+X right, +Y up, -Z forward)"
                ),
                "inference": {
                    "edge_threshold": kwargs["edge_threshold"],
                    "max_size": kwargs["max_size"],
                    "num_tokens": kwargs["num_tokens"],
                    "refine_steps": kwargs["refine_steps"],
                    "resolution_level": kwargs["resolution_level"],
                },
            }
            archive.writestr("metadata.json", json.dumps(metadata, indent=2))

        logger.info(
            "[%s] MoGe %sx%s refine_steps=%s",
            kwargs["filename"],
            width,
            height,
            kwargs["refine_steps"],
        )
        return archive_buffer.getvalue()


class RemoteMogeService:
    backend = "http"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._proxy = InferenceProxy(
            settings.reconstruct_url or "",
            timeout_s=settings.inference_timeout_s,
            name="MoGe",
        )
        self._ready = True

    def preload(self) -> None:
        return None

    def status(self) -> ServiceStatus:
        return ServiceStatus(
            enabled=True,
            backend=self.backend,
            ready=self._ready,
            detail=self.settings.reconstruct_url,
        )

    async def reconstruct_remote(
        self,
        raw: bytes,
        *,
        filename: str,
        content_type: str | None,
        include_mesh: bool,
        include_debug: bool,
        max_size: int,
        resolution_level: int,
        num_tokens: int | None,
        refine_steps: int,
        fov_x: float | None,
        edge_threshold: float,
    ) -> bytes:
        data = {
            "include_mesh": str(include_mesh).lower(),
            "include_debug": str(include_debug).lower(),
            "max_size": str(max_size),
            "resolution_level": str(resolution_level),
            "refine_steps": str(refine_steps),
            "edge_threshold": str(edge_threshold),
        }
        if num_tokens is not None:
            data["num_tokens"] = str(num_tokens)
        if fov_x is not None:
            data["fov_x"] = str(fov_x)
        result = await self._proxy.request(
            "POST",
            "/reconstruct",
            data=data,
            files={"file": (filename, raw, content_type or "application/octet-stream")},
        )
        return result.content


def build_moge_service(settings: Settings) -> MogeService | RemoteMogeService:
    if settings.reconstruct_url:
        return RemoteMogeService(settings)
    return MogeService(settings)
