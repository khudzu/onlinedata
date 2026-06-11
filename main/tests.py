from django.test import LiveServerTestCase, TestCase, tag
from django.urls import reverse
from selenium import webdriver

from main.crypto.mceliece_reed_muller import (
    decrypt_bits,
    decrypt_bytes,
    encrypt_bits,
    encrypt_bytes,
    generate_keypair,
)
from main.crypto.aes_reed_muller import (
    decrypt_image_bytes,
    decrypt_text,
    encrypt_image_bytes,
    encrypt_text,
    generate_aes_key,
    unwrap_aes_key,
    unwrap_aes_key_double,
    wrap_aes_key,
    wrap_aes_key_double,
)
from main.crypto.henon import (
    decrypt_image_bytes as henon_decrypt_image_bytes,
    decrypt_text as henon_decrypt_text,
    encrypt_image_bytes as henon_encrypt_image_bytes,
    encrypt_text as henon_encrypt_text,
    generate_henon_key,
)


@tag('functional')
class FunctionalTestCase(LiveServerTestCase):
    """Base class for functional test cases with selenium."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Change to another webdriver if desired (and update CI accordingly).
        options = webdriver.chrome.options.Options()
        # These options are needed for CI with Chromium.
        options.headless = True  # Disable GUI.
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        cls.selenium = webdriver.Chrome(options=options)

    @classmethod
    def tearDownClass(cls):
        cls.selenium.quit()
        super().tearDownClass()


class MainTestCase(TestCase):
    def test_root_url_status_200(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # You can also use path names instead of explicit paths.
        response = self.client.get(reverse('main:home'))
        self.assertEqual(response.status_code, 200)


class McElieceReedMullerTestCase(TestCase):
    def test_bit_round_trip(self):
        public_key, private_key = generate_keypair(order_m=4, seed=7)
        message = [1, 0, 1, 1, 0]

        ciphertext = encrypt_bits(message, public_key, seed=11)
        plaintext = decrypt_bits(ciphertext, private_key)

        self.assertEqual(plaintext.tolist(), message)

    def test_byte_round_trip(self):
        public_key, private_key = generate_keypair(order_m=4, seed=13)
        message = b"halo"

        ciphertext_blocks, padding = encrypt_bytes(message, public_key, seed=17)
        plaintext = decrypt_bytes(ciphertext_blocks, private_key, padding)

        self.assertEqual(plaintext, message)

    def test_aes_key_wrap_and_text_round_trip(self):
        aes_key = generate_aes_key()
        wrapped_key = wrap_aes_key(aes_key)
        unwrapped_key = unwrap_aes_key(wrapped_key)

        ciphertext = encrypt_text("rahasia", aes_key)
        plaintext = decrypt_text(ciphertext, unwrapped_key)

        self.assertEqual(unwrapped_key, aes_key)
        self.assertEqual(plaintext, "rahasia")

    def test_password_wrapped_aes_key_round_trip(self):
        aes_key = generate_aes_key()
        salt = "MDEyMzQ1Njc4OWFiY2RlZg=="

        wrapped_key = wrap_aes_key(aes_key, "kunci-user", salt)
        unwrapped_key = unwrap_aes_key(wrapped_key, "kunci-user", salt)

        self.assertEqual(unwrapped_key, aes_key)

    def test_double_wrapped_aes_key_round_trip(self):
        aes_key = generate_aes_key()

        wrapped_key = wrap_aes_key_double(aes_key)
        unwrapped_key = unwrap_aes_key_double(wrapped_key)

        self.assertEqual(unwrapped_key, aes_key)

    def test_aes_image_round_trip(self):
        aes_key = generate_aes_key()
        image_bytes = b"fake-image-bytes"

        ciphertext = encrypt_image_bytes(image_bytes, aes_key)
        plaintext = decrypt_image_bytes(ciphertext, aes_key)

        self.assertEqual(plaintext, image_bytes)

    def test_henon_text_round_trip(self):
        henon_key = generate_henon_key()

        ciphertext = henon_encrypt_text("rahasia", henon_key)
        plaintext = henon_decrypt_text(ciphertext, henon_key)

        self.assertEqual(plaintext, "rahasia")

    def test_henon_image_round_trip(self):
        import cv2
        import numpy as np

        henon_key = generate_henon_key()
        image = np.arange(48, dtype=np.uint8).reshape((4, 4, 3))
        ok, buffer = cv2.imencode(".png", image)
        self.assertTrue(ok)

        ciphertext = henon_encrypt_image_bytes(buffer.tobytes(), henon_key)
        plaintext = henon_decrypt_image_bytes(ciphertext, henon_key)
        restored = cv2.imdecode(np.frombuffer(plaintext, np.uint8), cv2.IMREAD_COLOR)

        self.assertTrue(np.array_equal(restored, image))


class MainFunctionalTestCase(FunctionalTestCase):
    def test_root_url_exists(self):
        self.selenium.get(f'{self.live_server_url}/')
        html = self.selenium.find_element_by_tag_name('html')
        self.assertNotIn('not found', html.text.lower())
        self.assertNotIn('error', html.text.lower())
