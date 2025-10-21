
"""
CAD rendering module using PyTorch3D with packed-batch support.

Features
- Load .OBJ (with materials) or .STL (via trimesh)
- Load a packed batch dict with keys: coord, faces, offset, offset_face
  * "offset" and "offset_face" are start indices per mesh (do NOT include 0)
- Orthographic or perspective camera
- Phong shading with point lights
- Background color control
- Render single or many meshes; save PNG or a tiled grid

Requirements
- pytorch
- pytorch3d
- Pillow
- numpy
- trimesh (only for .stl files)

Example (file input)
--------------------
python cad_renderer.py --input path/to/model.stl --out render.png --azim 45 --elev 30 --dist 2.5 --camera ortho

Example (packed batch)
----------------------
# batch = {
#   "coord": torch.randn(V, 3),
#   "faces": torch.randint(0, V, (F, 3)),
#   "offset": torch.tensor([v0, v1, ...]),         # starts per mesh (no 0)
#   "offset_face": torch.tensor([f0, f1, ...]),    # starts per mesh (no 0)
# }
# renderer = CADRenderer()
# renderer.load_from_packed_batch(batch)
# imgs = renderer.render_many()
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from pytorch3d.renderer import MeshRenderer
from pytorch3d.structures import Meshes
import os
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import numpy as np
from PIL import Image

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
    look_at_view_transform,
)
from pytorch3d.renderer import TexturesVertex


# -------------------------------
# Utilities
# -------------------------------

def _get_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _normalize_mesh(mesh: Meshes, unit_scale: float = 1.0) -> Meshes:
    """Center mesh at origin and scale so max extent == unit_scale."""
    verts = mesh.verts_list()

    for i in range(len(verts)):
        if verts[i].shape[0] > 0:
            vmin = verts[i].min(0).values
            vmax = verts[i].max(0).values
            center = (vmin + vmax) / 2.0
            extent = (vmax - vmin).max()
            scale = (unit_scale / extent) if float(extent) > 0 else 1.0
            verts[i] = (verts[i] - center) * scale

    mesh = Meshes(verts=verts, faces=mesh.faces_list(), textures=mesh.textures)
    return mesh


def _bounds_from_ends(total: int, ends: torch.Tensor) -> torch.Tensor:
    """Given end indices (len=B), return boundaries (len=B+1) with 0 prepended and total appended.
    Assumes ends do NOT include 0.
    """
    if ends.ndim != 1:
        raise ValueError("offsets must be 1-D end indices")
    if ends.numel() == 0:
        return torch.tensor([0, total], device=ends.device if ends.is_cuda else None, dtype=torch.long)
    dev = ends.device
    dt = ends.dtype
    # Ensure sorted (optional; remove if guaranteed sorted)
    if ends.numel() > 1 and not torch.all(ends[1:] >= ends[:-1]):
        ends, _ = torch.sort(ends)
    # Prepend 0 and append total
    bounds = torch.cat([
        torch.tensor([0], device=dev, dtype=dt),
        ends.to(dtype=dt),
    ])
    return bounds


# -------------------------------
# Data classes
# -------------------------------

@dataclass
class CameraConfig:
    type: str = "persp"  # "persp" or "ortho"
    azim: float = 0.0
    elev: float = 20.0
    dist: float = 2.5
    fov: float = 20.0  # for perspective & orthographic FoV
    at: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class LightConfig:
    location: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    ambient: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    diffuse: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    specular: Tuple[float, float, float] = (0.2, 0.2, 0.2)


# -------------------------------
# Main renderer
# -------------------------------

class CADRenderer:
    def __init__(
        self,
        image_size: int = 768,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        device: Optional[str] = None,
    ) -> None:
        self.device = _get_device(device)
        self.image_size = image_size
        self.background_color = background_color

        self._raster_settings = RasterizationSettings(
            image_size=image_size,
            faces_per_pixel=10,
            blur_radius=0.0,
        )
        self._blend_params = BlendParams(background_color=self.background_color)

        self.mesh: Optional[Meshes] = None
        self._renderer: Optional[MeshRenderer] = None

    # ------------- Loading from files -------------
    def load(
        self,
        path: str,
        color: Tuple[float, float, float] = (0.75, 0.75, 0.78),
        normalize: bool = True,
        unit_scale: float = 1.0
    ) -> None:
        """Load .obj or .stl into self.mesh on the selected device."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".obj":
            mesh = load_objs_as_meshes([path], device=self.device)
            # If the OBJ lacks textures, fall back to a uniform vertex color
            if not mesh.textures or mesh.textures.verts_features_list()[0] is None:
                verts = mesh.verts_packed()
                colors = torch.tensor(color, device=self.device, dtype=torch.float32).expand_as(verts)
                tex = TexturesVertex(verts_features=[colors])
                mesh = Meshes(verts=[mesh.verts_list()[0]], faces=[mesh.faces_list()[0]], textures=tex)
        elif ext == ".stl":
            try:
                import trimesh  # type: ignore
            except ImportError as e:
                raise ImportError("trimesh is required to load STL files: pip install trimesh") from e
            tm = trimesh.load(path, force='mesh')
            if tm.is_empty:
                raise ValueError("Failed to load STL: mesh is empty")
            verts = torch.tensor(np.asarray(tm.vertices), dtype=torch.float32, device=self.device)
            faces = torch.tensor(np.asarray(tm.faces), dtype=torch.int64, device=self.device)
            color_tensor = torch.tensor(color, device=self.device, dtype=torch.float32).repeat(verts.shape[0], 1)
            tex = TexturesVertex(verts_features=[color_tensor])
            mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Use .obj or .stl")

        if normalize:
            mesh = _normalize_mesh(mesh, unit_scale=unit_scale)

        self.mesh = mesh
        self._renderer = None

    # ------------- Loading from packed batch -------------
    def load_from_packed_batch(
        self,
        batch: dict,
        per_mesh_color: Optional[Tuple[float, float, float] | List[Tuple[float, float, float]]] = (0.75, 0.75, 0.78),
        normalize: bool = True,
        unit_scale: float = 1.0,
    ) -> None:
        """Create a Meshes from a packed batch dict.

        Expected keys in `batch`:
          - verts: (V_total, 3) float tensor of all vertices
          - faces: (F_total, 3) long tensor of all faces (indices into packed coord)
          - offset: vertex start indices per mesh (len=B), 0 NOT included
          - offset_face: face start indices per mesh (len=B), 0 NOT included
        """
        coord = batch["verts"].to(self.device)
        faces = batch["faces"].to(self.device)
        voff_ends = batch["vertex_offset"].to(self.device)
        foff_ends = batch["face_offset"].to(self.device)

        vbounds = _bounds_from_ends(coord.shape[0], voff_ends)
        fbounds = _bounds_from_ends(faces.shape[0], foff_ends)
        if vbounds.numel() != fbounds.numel():
            raise ValueError("Vertex and face splits mismatch (different B)")

        verts_list: List[torch.Tensor] = []
        faces_list: List[torch.Tensor] = []
        textures_list: List[torch.Tensor] = []
        B = vbounds.numel() - 1

        # Normalize color input to per-mesh list
        if isinstance(per_mesh_color, tuple):
            per_mesh_color = [per_mesh_color] * B
        elif per_mesh_color is None:
            per_mesh_color = [(0.75, 0.75, 0.78)] * B

        for i in range(B):
            vs, ve = int(vbounds[i].item()), int(vbounds[i + 1].item())
            fs, fe = int(fbounds[i].item()), int(fbounds[i + 1].item())
            v_chunk = coord[vs:ve]
            f_chunk = faces[fs:fe] - vs  # reindex faces to local verts

            verts_list.append(v_chunk)
            faces_list.append(f_chunk)

            color = torch.tensor(per_mesh_color[i], device=self.device, dtype=torch.float32)
            textures_list.append(color.expand(v_chunk.shape[0], 3))

        mesh = Meshes(
            verts=verts_list,
            faces=faces_list,
            textures=TexturesVertex(verts_features=textures_list),
        )

        if normalize:
            mesh = _normalize_mesh(mesh, unit_scale=unit_scale)

        self.mesh = mesh
        self._renderer = None

    # ------------- Cameras & Lights -------------
    def _make_cameras(self, cam: CameraConfig):
        R, T = look_at_view_transform(dist=cam.dist, elev=cam.elev, azim=cam.azim, at=torch.tensor([cam.at], device=self.device))
        if cam.type == "ortho":
            return FoVOrthographicCameras(device=self.device, R=R, T=T, znear=0.01, zfar=50.0, fov=cam.fov)
        elif cam.type == "persp":
            return FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=0.01, zfar=50.0, fov=cam.fov)
        else:
            raise ValueError("Camera type must be 'persp' or 'ortho'")

    def _make_lights(self, light: LightConfig):
        return PointLights(
            device=self.device,
            location=[light.location],
            ambient_color=[light.ambient],
            diffuse_color=[light.diffuse],
            specular_color=[light.specular],
        )

    def _get_renderer(self, cam: CameraConfig, light: LightConfig) -> MeshRenderer:
        cameras = self._make_cameras(cam)
        lights = self._make_lights(light)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self._raster_settings)
        shader = SoftPhongShader(device=self.device, cameras=cameras, lights=lights, blend_params=self._blend_params)
        return MeshRenderer(rasterizer=rasterizer, shader=shader)

    # ------------- Rendering -------------
    def render(
        self,
        cam: Optional[CameraConfig] = None,
        light: Optional[LightConfig] = None,
        return_alpha: bool = False,
    ) -> np.ndarray:
        """Render the current mesh and return an HxWx3 (or 4) uint8 image."""
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load(path) or load_from_packed_batch(batch) first.")
        cam = cam or CameraConfig()
        light = light or LightConfig()
        renderer = self._get_renderer(cam, light)
        with torch.no_grad():
            image = renderer(self.mesh, zfar=50.0)
        # image: (B, H, W, 4) RGBA float32 in [0,1]
        rgb = image[0, ..., :3].cpu().numpy()
        if return_alpha:
            a = image[0, ..., 3:4].cpu().numpy()
            out = np.concatenate([rgb, a], axis=-1)
        else:
            out = rgb
        out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return out

    def save_png(
        self,
        path: str,
        cam: Optional[CameraConfig] = None,
        light: Optional[LightConfig] = None,
        transparent: bool = False,
    ) -> None:
        img = self.render(cam=cam, light=light, return_alpha=transparent)
        mode = "RGBA" if transparent else "RGB"
        Image.fromarray(img, mode=mode).save(path)

    # ------------- Batch Rendering -------------
    def render_many(
        self,
        cam: Optional[CameraConfig] = None,
        light: Optional[LightConfig] = None,
        transparent: bool = False,
    ) -> List[np.ndarray]:
        """Render each mesh in a batched Meshes as separate images.
        If a single mesh is loaded, returns [image].
        """
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load(...) or load_from_packed_batch(...) first.")
        cam = cam or CameraConfig()
        light = light or LightConfig()
        renderer = self._get_renderer(cam, light)
        with torch.no_grad():
            images = renderer(self.mesh, zfar=50.0)
        # images: (B, H, W, 4)
        arr = images[..., :4 if transparent else 3].cpu().numpy()
        arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return [arr[i] for i in range(arr.shape[0])]

    def save_png_grid(
        self,
        path: str,
        ncols: int = 4,
        cam: Optional[CameraConfig] = None,
        light: Optional[LightConfig] = None,
        transparent: bool = False,
        pad: int = 4,
    ) -> None:
        """Render many and save as a tiled grid PNG."""
        imgs = self.render_many(cam=cam, light=light, transparent=transparent)
        if len(imgs) == 0:
            raise RuntimeError("Nothing to render")
        H, W = imgs[0].shape[:2]
        n = len(imgs)
        ncols = max(1, ncols)
        nrows = (n + ncols - 1) // ncols
        channels = imgs[0].shape[2]
        grid_h = nrows * H + (nrows - 1) * pad
        grid_w = ncols * W + (ncols - 1) * pad
        mode = "RGBA" if (transparent and channels == 4) else "RGB"
        bg = (0, 0, 0, 0) if mode == "RGBA" else tuple(int(c*255) for c in self.background_color)
        grid = Image.new(mode, (grid_w, grid_h), bg)
        for idx, im in enumerate(imgs):
            r = idx // ncols
            c = idx % ncols
            grid.paste(Image.fromarray(im), (c*(W+pad), r*(H+pad)))
        grid.save(path)


class RendererModule(nn.Module):
    def __init__(self, base: "CADRenderer"):
        super().__init__()
        # Store only config; re-create renderer per forward to handle per-batch cameras on device.
        self.base = base

    def forward(
        self,
        batch: dict,
        cam: Optional["CameraConfig"] = None,
        light: Optional["LightConfig"] = None,
        return_alpha: bool = True,
        normalize: bool = True,
        unit_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Inputs:
          batch: {coord, faces, offset, offset_face} (on any device; moved inside)
        Returns:
          images: (B, H, W, 4 or 3) float32 in [0,1] on the same device as inputs.
        """
        device = self.base.device
        # Build Meshes from packed batch (all ops are tensor ops -> differentiable)
        coord = batch["coord"].to(device)
        faces = batch["faces"].to(device)
        voff  = batch["offset"].to(device)
        foff  = batch["offset_face"].to(device)

        verts_splits = self.base._split_by_offsets(coord, voff)
        faces_splits = self.base._split_by_offsets(faces, foff)

        verts_list, faces_list = [], []
        for i, (v_chunk, f_chunk) in enumerate(zip(verts_splits, faces_splits)):
            base = 0 if i == 0 else sum(v.shape[0] for v in verts_splits[:i])
            verts_list.append(v_chunk)
            faces_list.append(f_chunk - base)

        # Simple gray per-vertex color (keeps autograd)
        colors = [torch.full((v.shape[0], 3), 0.75, device=device, dtype=coord.dtype) for v in verts_list]
        mesh = Meshes(verts=verts_list, faces=faces_list, textures=TexturesVertex(colors))

        if normalize:
            mesh = self.base._normalize_mesh(mesh, unit_scale=unit_scale)

        cam = cam or CameraConfig()
        light = light or LightConfig()
        renderer: MeshRenderer = self.base._get_renderer(cam, light)

        # Render -> (B, H, W, 4) float in [0,1]
        imgs = renderer(mesh, zfar=50.0)
        return imgs[..., :4] if return_alpha else imgs[..., :3]




# -------------------------------
# CLI usage
# -------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Render a CAD/mesh file with PyTorch3D")
    p.add_argument("--input", required=True, help="Path to .obj or .stl file")
    p.add_argument("--out", default="render.png", help="Output PNG path")
    p.add_argument("--size", type=int, default=768, help="Image size (square)")
    p.add_argument("--camera", choices=["persp", "ortho"], default="persp", help="Camera type")
    p.add_argument("--fov", type=float, default=20.0, help="Field of view")
    p.add_argument("--azim", type=float, default=30.0, help="Azimuth (deg)")
    p.add_argument("--elev", type=float, default=20.0, help="Elevation (deg)")
    p.add_argument("--dist", type=float, default=2.5, help="Camera distance")
    p.add_argument("--bg", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="Background RGB in [0,1]")
    p.add_argument("--color", type=float, nargs=3, default=(0.75, 0.75, 0.78), help="Fallback vertex color RGB in [0,1]")
    p.add_argument("--no-normalize", action="store_true", help="Disable unit normalization of mesh")
    p.add_argument("--unit-scale", type=float, default=1.0, help="Unit box size after normalization")
    return p.parse_args()


def _main():
    args = _parse_args()
    renderer = CADRenderer(image_size=args.size, background_color=tuple(args.bg))
    renderer.load(
        args.input,
        color=tuple(args.color),
        normalize=not args.no_normalize,
        unit_scale=args.unit_scale,
    )
    cam = CameraConfig(type=args.camera, azim=args.azim, elev=args.elev, dist=args.dist, fov=args.fov)
    renderer.save_png(args.out, cam=cam, transparent=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    _main()
