"""Serve the browser editor for the two manual scene-preparation steps.

Each step blocks until the page posts its result. The page shows the BEV mask
over RGB views of the Gaussians on the same grid. calibration.skip_manual
skips both steps and keeps a zero yaw and the automatic navigable map.
"""

import json
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlparse

import cv2
import numpy as np
from omegaconf import DictConfig

from one_scan_is_enough.scene import BevMap

logger = logging.getLogger(__name__)

STEPS = {"rotate": (1, "wall alignment"), "edit": (2, "navigable-area editing")}
RULE = "=" * 78
NOTICE = """
{rule}
MANUAL ANNOTATION REQUIRED - step {index}/2: {name}
The pipeline is paused until the result is saved in the browser editor:
    {url}

On a remote server, forward the port from the local machine first:
    ssh -N -L {port}:127.0.0.1:{port} <server>
and then open http://127.0.0.1:{port}/ in a local browser.
{rule}"""


def _encode(extension: str, image: np.ndarray, params: tuple = ()) -> bytes:
    success, encoded = cv2.imencode(extension, image, list(params))
    if not success:
        raise ValueError(f"Cannot encode {extension} image.")
    return encoded.tobytes()


def _parse_rotation(body: bytes, shape: tuple) -> dict:
    degrees = float(json.loads(body)["rotation_deg"])
    if not np.isfinite(degrees):
        raise ValueError("Rotation must be finite.")
    return {"yaw": float(np.deg2rad(degrees)), "summary": f"{degrees:.1f} deg"}


def _parse_mask(body: bytes, shape: tuple) -> dict:
    decoded = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None or decoded.shape != shape:
        raise ValueError("Map dimensions must stay unchanged.")
    navigable = (decoded > 127).astype(np.uint8) * 255
    return {"navigable": navigable, "summary": f"{np.count_nonzero(navigable)} navigable cells"}


def _serve(step: str, bev: BevMap, views: dict, config: DictConfig) -> dict:
    """Run one editing session and return the parsed result of its save request."""
    meta = {
        "step": step,
        "width": bev.navigable.shape[1],
        "height": bev.navigable.shape[0],
        "resolution": bev.resolution,
        "origin_xz": [float(value) for value in bev.origin_xz],
        "views": list(views),
    }
    assets = {
        "/": (
            files("one_scan_is_enough.preparation").joinpath("bev_editor.html").read_bytes(),
            "text/html; charset=utf-8",
        ),
        "/meta.json": (json.dumps(meta).encode(), "application/json"),
        "/mask.png": (_encode(".png", bev.navigable), "image/png"),
    }
    for name, image in views.items():
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        jpeg = _encode(".jpg", bgr, (cv2.IMWRITE_JPEG_QUALITY, 90))
        assets[f"/view/{name}.jpg"] = (jpeg, "image/jpeg")
    parse = _parse_rotation if step == "rotate" else _parse_mask
    result = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            asset = assets.get(urlparse(self.path).path)
            if asset is None:
                self.send_error(404)
                return
            data, content_type = asset
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if urlparse(self.path).path != "/save":
                self.send_error(404)
                return
            if result:
                self.send_error(409, "This step is already saved.")
                return
            body = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                result.update(parse(body, bev.navigable.shape))
            except (KeyError, ValueError) as error:
                self.send_error(400, str(error))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            threading.Thread(target=server.shutdown, daemon=True).start()

        def log_message(self, fmt, *args):
            return

    index, name = STEPS[step]
    with ThreadingHTTPServer((config.host, config.port), Handler) as server:
        port = server.server_port
        url = f"http://{config.host}:{port}/"
        logger.warning(NOTICE.format(rule=RULE, index=index, name=name, url=url, port=port))
        # The page of the first step moves on to the second by itself.
        if config.open_browser and index == 1:
            webbrowser.open(url)
        server.serve_forever()
    logger.info("Saved manual %s: %s", name, result["summary"])
    return result


def _skip(step: str) -> None:
    index, name = STEPS[step]
    logger.warning(
        "Skipping manual step %s/2 (%s) because calibration.skip_manual=true; "
        "the automatic result is kept.",
        index,
        name,
    )


def calibrate_yaw(bev: BevMap, views: dict, config: DictConfig) -> float:
    """Wait for manual wall alignment and return yaw in radians."""
    if config.skip_manual:
        _skip("rotate")
        return 0.0
    return _serve("rotate", bev, views, config)["yaw"]


def edit_navigable(bev: BevMap, views: dict, config: DictConfig) -> np.ndarray:
    """Wait for manual map edits and return the confirmed binary array."""
    if config.skip_manual:
        _skip("edit")
        return bev.navigable
    return _serve("edit", bev, views, config)["navigable"]
