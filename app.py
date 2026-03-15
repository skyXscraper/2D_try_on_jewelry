# ---------------------------------------------------------------------------
# app.py — Flask server: seller upload + dashboard, customer try-on
# ---------------------------------------------------------------------------

import os, uuid, json, socket
from datetime import datetime

import cv2
import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, Response, send_file, jsonify)

import config
from landmarks import get_face_landmarks
from overlay   import load_overlay, overlay_image, split_pair
from smoother  import PositionSmoother
from preprocess import remove_bg
from qr_generator import generate_qr

app = Flask(__name__)

SESSIONS_DIR  = os.path.join(os.path.dirname(__file__), "sessions")
os.makedirs(SESSIONS_DIR,       exist_ok=True)
os.makedirs(config.PROCESSED_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _save_product(pid: str, data: dict):
    with open(os.path.join(SESSIONS_DIR, f"{pid}.json"), "w") as f:
        json.dump(data, f, indent=2)


def _load_product(pid: str) -> dict | None:
    path = os.path.join(SESSIONS_DIR, f"{pid}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _all_products() -> list[dict]:
    products = []
    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(SESSIONS_DIR, fname)) as f:
                    products.append(json.load(f))
            except Exception:
                pass
    return products


# ---------------------------------------------------------------------------
# Overlay helper — shared by MJPEG stream and static-image try-on
# ---------------------------------------------------------------------------

def _apply_overlay(frame: np.ndarray, product: dict, data: dict,
                   smoothers: dict | None = None):
    """Apply jewelry overlay for *product* onto *frame* (in-place)."""
    ptype = product["type"]
    img   = load_overlay(product["processed"])
    if img is None:
        return

    fw, tilt = data["face_width"], data["tilt_angle"]

    if ptype == "earring_pair":
        el, er = split_pair(img)
        drop = int(fw * config.EARRING_Y_OFFSET_RATIO)
        out  = int(fw * config.EARRING_X_OUTWARD_RATIO)
        size = int(fw * config.SCALE_FACTOR_EARRING)
        lx, ly = data["left_ear"]
        rx, ry = data["right_ear"]
        if smoothers:
            lx, ly, size_l = smoothers["l"].smooth(lx - out, ly + drop, size)
            rx, ry, size_r = smoothers["r"].smooth(rx + out, ry + drop, size)
        else:
            lx, ly, size_l = lx - out, ly + drop, size
            rx, ry, size_r = rx + out, ry + drop, size
        overlay_image(frame, el, int(lx), int(ly), int(size_l), tilt)
        overlay_image(frame, er, int(rx), int(ry), int(size_r), tilt)

    elif ptype == "necklace":
        size = int(fw * config.SCALE_FACTOR_NECKLACE)
        nx, ny = data["jaw_mid"]
        drop = int(fw * config.NECKLACE_Y_OFFSET_RATIO)
        if smoothers:
            nx, ny, size = smoothers["n"].smooth(nx, ny + drop, size)
        else:
            ny += drop
        overlay_image(frame, img, int(nx), int(ny), int(size), tilt)

    elif ptype == "spectacles":
        le = data.get("left_eye_outer")
        re = data.get("right_eye_outer")
        if le and re:
            cx = (le[0] + re[0]) // 2
            cy = (le[1] + re[1]) // 2
            eye_span = data.get("eye_span", fw * 0.45)
            size = int(eye_span * 1.15)
            if smoothers:
                cx, cy, size = smoothers["n"].smooth(cx, cy, size)
            overlay_image(frame, img, int(cx), int(cy), int(size), tilt)


# ---------------------------------------------------------------------------
# Webcam helper
# ---------------------------------------------------------------------------

def _open_cam():
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, f = cap.read()
            if ok and f is not None:
                return cap
            cap.release()
    return None


def _gen_frames(product: dict):
    cap = _open_cam()
    if cap is None:
        return
    sm = {"l": PositionSmoother(), "r": PositionSmoother(), "n": PositionSmoother()}
    fail = 0
    try:
        while True:
            try:
                ok, frame = cap.read()
            except cv2.error:
                fail += 1
                if fail > 10: break
                continue
            if not ok or frame is None:
                fail += 1
                if fail > 10: break
                continue
            fail = 0
            frame = cv2.flip(frame, 1)
            data  = get_face_landmarks(frame)
            if data:
                _apply_overlay(frame, product, data, sm)
            else:
                for s in sm.values(): s.reset()
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Routes — Seller
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("seller"))


@app.route("/seller")
def seller():
    return render_template("upload.html")


@app.route("/seller/upload", methods=["POST"])
def upload():
    base_url = f"http://{_local_ip()}:5000"
    product_fields = {
        "earring_pair": ("earring_pair.png", "Earring Pair"),
        "necklace":     ("necklace.png",     "Necklace"),
        "spectacles":   ("spectacles.png",   "Spectacles"),
    }
    created = 0
    for field, (filename, label) in product_fields.items():
        f = request.files.get(field)
        if not f or not f.filename:
            continue

        pid      = str(uuid.uuid4())[:8]
        prod_dir = os.path.join(config.PROCESSED_DIR, pid)
        os.makedirs(prod_dir, exist_ok=True)

        raw_path = os.path.join(prod_dir, f"raw_{filename}")
        f.save(raw_path)

        processed_path = remove_bg(raw_path, prod_dir)

        tryon_url = f"{base_url}/tryon/{pid}"
        qr_path   = os.path.join(prod_dir, "qr.png")
        generate_qr(tryon_url, qr_path)

        _save_product(pid, {
            "id":        pid,
            "type":      field,
            "label":     label,
            "name":      f.filename,
            "processed": processed_path,
            "qr":        qr_path,
            "tryon_url": tryon_url,
            "created":   datetime.now().isoformat(),
        })
        created += 1

    if created == 0:
        return "No files uploaded", 400
    return redirect(url_for("dashboard"))


@app.route("/seller/dashboard")
def dashboard():
    return render_template("dashboard.html", products=_all_products())


# ---------------------------------------------------------------------------
# Routes — Customer
# ---------------------------------------------------------------------------

@app.route("/tryon/<pid>")
def tryon(pid):
    product = _load_product(pid)
    if not product:
        return "Session not found", 404
    return render_template("tryon.html", product=product)


@app.route("/stream/<pid>")
def stream(pid):
    product = _load_product(pid)
    if not product:
        return "Not found", 404
    return Response(_gen_frames(product),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/tryon-image/<pid>", methods=["POST"])
def tryon_image(pid):
    product = _load_product(pid)
    if not product:
        return jsonify(error="Not found"), 404
    f = request.files.get("face")
    if not f:
        return jsonify(error="No face image provided"), 400

    buf   = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="Cannot decode image"), 400

    data = get_face_landmarks(frame)
    if data:
        _apply_overlay(frame, product, data)

    result_path = os.path.join(config.PROCESSED_DIR, pid, "result.jpg")
    cv2.imwrite(result_path, frame)
    return jsonify(result_url=f"/preview/{pid}")


@app.route("/preview/<pid>")
def preview(pid):
    result_path = os.path.join(config.PROCESSED_DIR, pid, "result.jpg")
    if not os.path.exists(result_path):
        return "No result yet", 404
    return send_file(result_path, mimetype="image/jpeg")


@app.route("/download/<pid>")
def download(pid):
    result_path = os.path.join(config.PROCESSED_DIR, pid, "result.jpg")
    if not os.path.exists(result_path):
        return "No result yet", 404
    return send_file(result_path, as_attachment=True,
                     download_name="tryon_result.jpg")


@app.route("/product-image/<pid>")
def product_image(pid):
    product = _load_product(pid)
    if not product:
        return "Not found", 404
    return send_file(product["processed"], mimetype="image/png")


@app.route("/qr/<pid>")
def qr_image(pid):
    product = _load_product(pid)
    if not product:
        return "Not found", 404
    return send_file(product["qr"], mimetype="image/png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ip = _local_ip()
    print(f"\n  Seller → http://{ip}:5000/seller")
    print(f"  Dashboard → http://{ip}:5000/seller/dashboard\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
