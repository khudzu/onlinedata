import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from django.conf import settings

from main.crypto.henon import henon_encrypt_image


STATIC_IMAGE_DIR = Path(__file__).resolve().parent.parent / "static" / "img"
IMAGE_CANDIDATES = [
    STATIC_IMAGE_DIR / "lena.png",
    Path(__file__).resolve().parent.parent / "images" / "lena.png",
]


def summarize(values):
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def mse(original, encrypted):
    diff = original.astype(np.float64) - encrypted.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(mse_value):
    if mse_value == 0:
        return None
    return float(10 * np.log10((255.0 * 255.0) / mse_value))


def ssim_channel(original, encrypted):
    original = original.astype(np.float64)
    encrypted = encrypted.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = float(np.mean(original))
    mu_y = float(np.mean(encrypted))
    var_x = float(np.var(original))
    var_y = float(np.var(encrypted))
    cov_xy = float(np.mean((original - mu_x) * (encrypted - mu_y)))
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float(numerator / denominator)


def ssim(original, encrypted):
    return float(np.mean([ssim_channel(original[:, :, i], encrypted[:, :, i]) for i in range(original.shape[2])]))


def hill_encrypt(img, a=2, b=3):
    mod = 256
    rows, cols, channels = img.shape
    working = img
    if cols % 2 == 1:
        working = cv2.copyMakeBorder(working, 0, 0, 0, 1, cv2.BORDER_REPLICATE)
        rows, cols, channels = working.shape

    key = np.array([[1, a], [b, a * b + 1]], dtype=np.int64)
    encrypted = np.zeros((rows, cols, channels), dtype=np.uint8)
    for x in range(rows):
        for y in range(0, cols, 2):
            block = working[x, y : y + 2, :].astype(np.int64)
            encrypted[x, y : y + 2, :] = (key @ (block % mod)) % mod
    return encrypted[:, : img.shape[1], :].astype(np.uint8)


def arnold_cat_map_encrypt(img, a=2, b=3, iterations=2):
    rows, cols, channels = img.shape
    if rows != cols:
        side = min(rows, cols)
        img = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
        rows = cols = side

    key = np.array([[1, a], [b, a * b + 1]], dtype=np.int64)
    encrypted = img.copy()
    for _ in range(iterations):
        mapped = np.zeros((rows, cols, channels), dtype=np.uint8)
        for x in range(rows):
            for y in range(cols):
                source = key @ np.array([x, y], dtype=np.int64) % rows
                mapped[x, y] = encrypted[source[0], source[1]]
        encrypted = mapped
    return encrypted.astype(np.uint8)


def hill_arnold_encrypt(img):
    return arnold_cat_map_encrypt(hill_encrypt(img))


def bytes_to_image(data, shape):
    needed = int(np.prod(shape))
    values = np.frombuffer(data, dtype=np.uint8)
    if values.size < needed:
        values = np.pad(values, (0, needed - values.size), mode="constant")
    else:
        values = values[:needed]
    return values.reshape(shape).astype(np.uint8)


def aes_gcm_raw_pixels(img):
    key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, img.tobytes(), None)
    return bytes_to_image(ciphertext, img.shape)


def henon_map_encrypt(img):
    return henon_encrypt_image(img, a=5, b=7)


def find_benchmark_image():
    for image_path in IMAGE_CANDIDATES:
        if image_path.exists():
            return image_path
    raise FileNotFoundError("No benchmark image found.")


def run_image_encryption_benchmark(runs=3):
    runs = max(1, min(int(runs), 10))
    image_path = find_benchmark_image()
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Could not read benchmark image: {image_path}")

    algorithms = {
        "hill_cipher_then_arnold_cat_map": hill_arnold_encrypt,
        "aes_gcm_raw_pixels": aes_gcm_raw_pixels,
        "henon_map": henon_map_encrypt,
    }
    results = []
    started_at = time.perf_counter()

    for name, encrypt_fn in algorithms.items():
        timings = []
        encrypted = None
        for _ in range(runs):
            start = time.perf_counter()
            encrypted = encrypt_fn(original)
            timings.append((time.perf_counter() - start) * 1000)
        mse_value = mse(original, encrypted)
        results.append(
            {
                "algorithm": name,
                "runs": runs,
                "encrypt_time": summarize(timings),
                "mse": mse_value,
                "psnr_db": psnr(mse_value),
                "ssim": ssim(original, encrypted),
            }
        )

    return {
        "environment": {
            "database_engine": settings.DATABASES["default"]["ENGINE"],
            "image": str(image_path),
            "image_shape": list(original.shape),
            "total_runtime_ms": (time.perf_counter() - started_at) * 1000,
            "note": "Timing was measured on the running server environment.",
        },
        "results": results,
    }
