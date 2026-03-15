# ---------------------------------------------------------------------------
# qr_generator.py — QR code generation
# ---------------------------------------------------------------------------

import qrcode
from PIL import Image


def generate_qr(url: str, out_path: str) -> str:
    """Generate a QR code for *url* and save to *out_path*. Returns out_path."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1a1a", back_color="white")
    img.save(out_path)
    return out_path
