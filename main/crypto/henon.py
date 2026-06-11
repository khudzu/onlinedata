import base64
import json

import cv2
import numpy as np


DEFAULT_A = 5
DEFAULT_B = 7


def _b64encode(data):
    return base64.urlsafe_b64encode(data).decode("ascii")


def _b64decode(data):
    return base64.urlsafe_b64decode(data.encode("ascii"))


def _modinv(value, modulus=256):
    value = int(value)
    for candidate in range(modulus):
        if (value * candidate) % modulus == 1:
            return candidate
    raise ValueError("Henon parameter b must be invertible modulo 256.")


def generate_henon_key(a=DEFAULT_A, b=DEFAULT_B):
    _modinv(b)
    return json.dumps(
        {
            "alg": "Henon-Map-Block2",
            "a": int(a),
            "b": int(b),
        },
        separators=(",", ":"),
    )


def is_henon_key(payload):
    try:
        return json.loads(payload).get("alg") == "Henon-Map-Block2"
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _load_key(payload):
    data = json.loads(payload)
    if data.get("alg") != "Henon-Map-Block2":
        raise ValueError("Not a Henon key payload.")
    return int(data.get("a", DEFAULT_A)), int(data.get("b", DEFAULT_B))


def _encrypt_byte_pairs(data, a, b):
    values = np.frombuffer(data, dtype=np.uint8)
    padding = 0
    if values.size % 2 == 1:
        values = np.pad(values, (0, 1), mode="constant")
        padding = 1

    pairs = values.reshape(-1, 2).astype(np.int64)
    x = pairs[:, 0]
    y = pairs[:, 1]
    encrypted = np.empty_like(pairs, dtype=np.uint8)
    encrypted[:, 0] = (1 - a * x * x + y) % 256
    encrypted[:, 1] = (b * x) % 256
    return encrypted.reshape(-1).tobytes(), padding


def _decrypt_byte_pairs(data, a, b, padding=0):
    values = np.frombuffer(data, dtype=np.uint8)
    if values.size % 2 == 1:
        raise ValueError("Henon ciphertext length must be even.")

    b_inv = _modinv(b)
    pairs = values.reshape(-1, 2).astype(np.int64)
    x_new = pairs[:, 0]
    y_new = pairs[:, 1]
    decrypted = np.empty_like(pairs, dtype=np.uint8)
    x = (b_inv * y_new) % 256
    y = (x_new - 1 + a * x * x) % 256
    decrypted[:, 0] = x
    decrypted[:, 1] = y

    plain = decrypted.reshape(-1)
    if padding:
        plain = plain[:-int(padding)]
    return plain.tobytes()


def henon_encrypt_image(img, a=DEFAULT_A, b=DEFAULT_B):
    img = img.astype(np.uint8)
    rows, cols, channels = img.shape
    cipher = np.zeros_like(img)

    for row in range(rows):
        for col in range(0, cols - 1, 2):
            for channel in range(channels):
                x = int(img[row, col, channel])
                y = int(img[row, col + 1, channel])
                cipher[row, col, channel] = (1 - a * x * x + y) % 256
                cipher[row, col + 1, channel] = (b * x) % 256

    if cols % 2 == 1:
        cipher[:, -1, :] = img[:, -1, :]
    return cipher.astype(np.uint8)


def henon_decrypt_image(cipher, a=DEFAULT_A, b=DEFAULT_B):
    cipher = cipher.astype(np.uint8)
    rows, cols, channels = cipher.shape
    plain = np.zeros_like(cipher)
    b_inv = _modinv(b)

    for row in range(rows):
        for col in range(0, cols - 1, 2):
            for channel in range(channels):
                x_new = int(cipher[row, col, channel])
                y_new = int(cipher[row, col + 1, channel])
                x = (b_inv * y_new) % 256
                y = (x_new - 1 + a * x * x) % 256
                plain[row, col, channel] = x
                plain[row, col + 1, channel] = y

    if cols % 2 == 1:
        plain[:, -1, :] = cipher[:, -1, :]
    return plain.astype(np.uint8)


def encrypt_text(plaintext, key_payload):
    a, b = _load_key(key_payload)
    ciphertext, padding = _encrypt_byte_pairs(str(plaintext).encode("utf-8"), a, b)
    return json.dumps(
        {
            "alg": "Henon-Map-Block2-Text",
            "padding": padding,
            "ciphertext": _b64encode(ciphertext),
        },
        separators=(",", ":"),
    )


def decrypt_text(payload, key_payload):
    try:
        data = json.loads(payload)
        if data.get("alg") != "Henon-Map-Block2-Text":
            return payload
        a, b = _load_key(key_payload)
        plaintext = _decrypt_byte_pairs(
            _b64decode(data["ciphertext"]),
            a,
            b,
            data.get("padding", 0),
        )
        return plaintext.decode("utf-8")
    except (TypeError, ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return payload


def encrypt_image_bytes(image_bytes, key_payload):
    a, b = _load_key(key_payload)
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image bytes could not be decoded.")

    encrypted_image = henon_encrypt_image(image, a, b)
    ok, buffer = cv2.imencode(".png", encrypted_image)
    if not ok:
        raise ValueError("Encrypted image could not be encoded.")

    return json.dumps(
        {
            "alg": "Henon-Map-Block2-Image",
            "format": "png",
            "ciphertext": _b64encode(buffer.tobytes()),
        },
        separators=(",", ":"),
    ).encode("utf-8")


def decrypt_image_bytes(payload, key_payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    data = json.loads(payload)
    if data.get("alg") != "Henon-Map-Block2-Image":
        raise ValueError("Not a Henon image payload.")

    a, b = _load_key(key_payload)
    encrypted_png = _b64decode(data["ciphertext"])
    encrypted_image = cv2.imdecode(np.frombuffer(encrypted_png, np.uint8), cv2.IMREAD_COLOR)
    if encrypted_image is None:
        raise ValueError("Encrypted image could not be decoded.")

    decrypted_image = henon_decrypt_image(encrypted_image, a, b)
    ok, buffer = cv2.imencode(".png", decrypted_image)
    if not ok:
        raise ValueError("Decrypted image could not be encoded.")
    return buffer.tobytes()


def get_payload_ciphertext_bytes(payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    data = json.loads(payload)
    return _b64decode(data["ciphertext"])


def get_payload_ciphertext_text(payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    try:
        data = json.loads(payload)
        return data["ciphertext"]
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return payload
