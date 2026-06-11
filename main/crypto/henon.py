import base64
import json
import secrets

import cv2
import numpy as np


DEFAULT_A = 5
DEFAULT_B = 7
LEGACY_ALG = "Henon-Map-Block2"
SECURE_ALG = "Henon-Map-Permutation-XOR-v2"
HILL_HENON_ALG = "Hill-Cipher-Henon-Map"
HILL_HENON_TEXT_ALG = "Hill-Cipher-Henon-Map-Text"
HILL_HENON_IMAGE_ALG = "Hill-Cipher-Henon-Map-Image"
DEFAULT_CHAOS_A = 1.4
DEFAULT_CHAOS_B = 0.3


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
            "alg": HILL_HENON_ALG,
            "hill_a": 2,
            "hill_b": 3,
            "chaos_a": DEFAULT_CHAOS_A,
            "chaos_b": DEFAULT_CHAOS_B,
            "x0": round((secrets.randbelow(800000) + 100000) / 1000000, 6),
            "y0": round((secrets.randbelow(800000) + 100000) / 1000000, 6),
        },
        separators=(",", ":"),
    )


def is_henon_key(payload):
    try:
        return json.loads(payload).get("alg") in {LEGACY_ALG, SECURE_ALG, HILL_HENON_ALG}
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _load_key_data(payload):
    data = json.loads(payload)
    if data.get("alg") not in {LEGACY_ALG, SECURE_ALG, HILL_HENON_ALG}:
        raise ValueError("Not a Henon key payload.")
    return data


def _load_block_key(payload):
    data = _load_key_data(payload)
    return (
        int(data.get("block_a", data.get("a", DEFAULT_A))),
        int(data.get("block_b", data.get("b", DEFAULT_B))),
    )


def _load_secure_key(payload):
    data = _load_key_data(payload)
    return {
        "alg": data.get("alg"),
        "block_a": int(data.get("block_a", data.get("a", DEFAULT_A))),
        "block_b": int(data.get("block_b", data.get("b", DEFAULT_B))),
        "hill_a": int(data.get("hill_a", 2)),
        "hill_b": int(data.get("hill_b", 3)),
        "chaos_a": float(data.get("chaos_a", DEFAULT_CHAOS_A)),
        "chaos_b": float(data.get("chaos_b", DEFAULT_CHAOS_B)),
        "x0": float(data.get("x0", 0.314159)),
        "y0": float(data.get("y0", 0.271828)),
    }


def _henon_sequence(length, x0, y0, a=DEFAULT_CHAOS_A, b=DEFAULT_CHAOS_B):
    sequence = np.empty(length, dtype=np.float64)
    x = float(x0)
    y = float(y0)
    warmup = 128
    index = 0

    for step in range(length + warmup):
        x_new = 1.0 - a * x * x + y
        y_new = b * x
        x, y = x_new, y_new
        if not np.isfinite(x) or not np.isfinite(y):
            x = (float(x0) % 1.0) + 0.123457
            y = (float(y0) % 1.0) + 0.765431
        if step >= warmup:
            sequence[index] = x + (0.5 * y)
            index += 1

    return sequence


def _keystream_and_permutation(length, key):
    sequence = _henon_sequence(
        length * 2,
        key["x0"],
        key["y0"],
        key["chaos_a"],
        key["chaos_b"],
    )
    stream_source = sequence[:length]
    permutation_source = sequence[length:]
    keystream = (np.floor(np.abs(np.sin(stream_source) * 10**14)) % 256).astype(np.uint8)
    permutation = np.argsort(permutation_source, kind="mergesort")
    return keystream, permutation


def _henon_permutation(length, key):
    sequence = _henon_sequence(
        length,
        key["x0"],
        key["y0"],
        key["chaos_a"],
        key["chaos_b"],
    )
    return np.argsort(sequence, kind="mergesort")


def _hill_matrix(a, b):
    return np.array([[1, a], [b, a * b + 1]], dtype=np.int64)


def _hill_inverse_matrix(a, b):
    return np.array([[a * b + 1, -a], [-b, 1]], dtype=np.int64) % 256


def _hill_transform_bytes(data, matrix, padding=0):
    values = np.frombuffer(data, dtype=np.uint8)
    added_padding = 0
    if values.size % 2 == 1:
        values = np.pad(values, (0, 1), mode="constant")
        added_padding = 1

    pairs = values.reshape(-1, 2).astype(np.int64)
    transformed = (pairs @ matrix.T) % 256
    flattened = transformed.astype(np.uint8).reshape(-1)
    if padding:
        flattened = flattened[:-int(padding)]
    return flattened.tobytes(), added_padding


def _hill_encrypt_bytes(data, a, b):
    return _hill_transform_bytes(data, _hill_matrix(a, b))


def _hill_decrypt_bytes(data, a, b, padding=0):
    plain, _ = _hill_transform_bytes(data, _hill_inverse_matrix(a, b), padding)
    return plain


def _encrypt_image_secure(img, key):
    base_cipher = henon_encrypt_image(img, key["block_a"], key["block_b"])
    flat = base_cipher.reshape(-1)
    keystream, permutation = _keystream_and_permutation(flat.size, key)
    cipher_flat = np.bitwise_xor(flat[permutation], keystream)
    return cipher_flat.reshape(base_cipher.shape).astype(np.uint8)


def _decrypt_image_secure(cipher, key):
    flat = cipher.reshape(-1)
    keystream, permutation = _keystream_and_permutation(flat.size, key)
    shuffled = np.bitwise_xor(flat, keystream)
    base_flat = np.empty_like(shuffled)
    base_flat[permutation] = shuffled
    base_cipher = base_flat.reshape(cipher.shape).astype(np.uint8)
    return henon_decrypt_image(base_cipher, key["block_a"], key["block_b"])


def _encrypt_image_hill_henon(img, key):
    flat = img.reshape(-1)
    if flat.size % 2 == 1:
        hill_bytes, _ = _hill_encrypt_bytes(flat[:-1].tobytes(), key["hill_a"], key["hill_b"])
        hill_flat = np.concatenate([np.frombuffer(hill_bytes, dtype=np.uint8), flat[-1:]])
    else:
        hill_bytes, _ = _hill_encrypt_bytes(flat.tobytes(), key["hill_a"], key["hill_b"])
        hill_flat = np.frombuffer(hill_bytes, dtype=np.uint8)
    permutation = _henon_permutation(hill_flat.size, key)
    cipher_flat = hill_flat[permutation]
    return cipher_flat.reshape(img.shape).astype(np.uint8)


def _decrypt_image_hill_henon(cipher, key):
    flat = cipher.reshape(-1)
    permutation = _henon_permutation(flat.size, key)
    hill_flat = np.empty_like(flat)
    hill_flat[permutation] = flat
    if hill_flat.size % 2 == 1:
        plain_bytes = _hill_decrypt_bytes(
            hill_flat[:-1].tobytes(),
            key["hill_a"],
            key["hill_b"],
        )
        plain_flat = np.concatenate([np.frombuffer(plain_bytes, dtype=np.uint8), hill_flat[-1:]])
    else:
        plain_bytes = _hill_decrypt_bytes(
            hill_flat.tobytes(),
            key["hill_a"],
            key["hill_b"],
        )
        plain_flat = np.frombuffer(plain_bytes, dtype=np.uint8)
    return plain_flat.reshape(cipher.shape).astype(np.uint8)


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
    key = _load_secure_key(key_payload)
    if key["alg"] == HILL_HENON_ALG:
        ciphertext, padding = _hill_encrypt_bytes(
            str(plaintext).encode("utf-8"),
            key["hill_a"],
            key["hill_b"],
        )
        algorithm = HILL_HENON_TEXT_ALG
    else:
        a, b = _load_block_key(key_payload)
        ciphertext, padding = _encrypt_byte_pairs(str(plaintext).encode("utf-8"), a, b)
        algorithm = "Henon-Map-Block2-Text"
    return json.dumps(
        {
            "alg": algorithm,
            "padding": padding,
            "ciphertext": _b64encode(ciphertext),
        },
        separators=(",", ":"),
    )


def decrypt_text(payload, key_payload):
    try:
        data = json.loads(payload)
        if data.get("alg") == HILL_HENON_TEXT_ALG:
            key = _load_secure_key(key_payload)
            plaintext = _hill_decrypt_bytes(
                _b64decode(data["ciphertext"]),
                key["hill_a"],
                key["hill_b"],
                data.get("padding", 0),
            )
            return plaintext.decode("utf-8")
        if data.get("alg") != "Henon-Map-Block2-Text":
            return payload
        a, b = _load_block_key(key_payload)
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
    key = _load_secure_key(key_payload)
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image bytes could not be decoded.")

    if key["alg"] == HILL_HENON_ALG:
        encrypted_image = _encrypt_image_hill_henon(image, key)
    elif key["alg"] == SECURE_ALG:
        encrypted_image = _encrypt_image_secure(image, key)
    else:
        encrypted_image = henon_encrypt_image(image, key["block_a"], key["block_b"])
    ok, buffer = cv2.imencode(".png", encrypted_image)
    if not ok:
        raise ValueError("Encrypted image could not be encoded.")

    return json.dumps(
        {
            "alg": HILL_HENON_IMAGE_ALG if key["alg"] == HILL_HENON_ALG else "Henon-Map-Block2-Image",
            "format": "png",
            "ciphertext": _b64encode(buffer.tobytes()),
        },
        separators=(",", ":"),
    ).encode("utf-8")


def decrypt_image_bytes(payload, key_payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    data = json.loads(payload)
    if data.get("alg") not in {"Henon-Map-Block2-Image", HILL_HENON_IMAGE_ALG}:
        raise ValueError("Not a Henon image payload.")

    key = _load_secure_key(key_payload)
    encrypted_png = _b64decode(data["ciphertext"])
    encrypted_image = cv2.imdecode(np.frombuffer(encrypted_png, np.uint8), cv2.IMREAD_COLOR)
    if encrypted_image is None:
        raise ValueError("Encrypted image could not be decoded.")

    if key["alg"] == HILL_HENON_ALG:
        decrypted_image = _decrypt_image_hill_henon(encrypted_image, key)
    elif key["alg"] == SECURE_ALG:
        decrypted_image = _decrypt_image_secure(encrypted_image, key)
    else:
        decrypted_image = henon_decrypt_image(encrypted_image, key["block_a"], key["block_b"])
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
