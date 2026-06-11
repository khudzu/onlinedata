from django.conf import settings
from django.http import FileResponse, Http404
from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.decorators import user_passes_test
from main.functions.functions import handle_uploaded_file
from main.crypto.aes_reed_muller import (
	decrypt_image_bytes as aes_decrypt_image_bytes,
	decrypt_text as aes_decrypt_text,
	get_payload_ciphertext_bytes,
	get_payload_ciphertext_text,
	unwrap_aes_key,
)
from main.crypto.henon import (
	decrypt_image_bytes as henon_decrypt_image_bytes,
	decrypt_text as henon_decrypt_text,
	encrypt_image_bytes as henon_encrypt_image_bytes,
	encrypt_text as henon_encrypt_text,
	generate_henon_key,
	is_henon_key,
)

# Create your views here.

from .forms import DecryptionKeyForm, PostForm, LoginForm
from .benchmark import run_wrapping_benchmark
from .image_benchmark import run_image_encryption_benchmark
from .models import PostModel
from sympy import *
import cv2
import json
import numpy as np
from pathlib import Path
from io import BytesIO
from time import perf_counter
from urllib.parse import quote
from uuid import uuid4

STATIC_IMAGE_DIR = Path(__file__).resolve().parent.parent / 'static' / 'img'
ENCRYPTED_IMAGE_DIR = settings.MEDIA_ROOT / 'encrypted_images'


def get_image_url(image_name):
	filename = Path(str(image_name)).name
	if not Path(filename).suffix:
		filename = f'{filename}.png'
	return f'/encrypted-images/{quote(filename)}'


def get_decrypted_image_url(image_name):
	filename = Path(str(image_name)).name
	if not Path(filename).suffix:
		filename = f'{filename}.png'
	return f'/decrypted-images/{quote(filename)}'


def find_stored_image(filename):
	if not Path(filename).suffix:
		filename = f'{filename}.png'

	for image_dir in [ENCRYPTED_IMAGE_DIR, STATIC_IMAGE_DIR]:
		image_path = image_dir / filename
		if image_path.exists():
			return image_path

	return None


def visible_posts_for(user):
	if user.is_superuser:
		return PostModel.objects.all()
	if not user.is_authenticated:
		return PostModel.objects.none()
	return PostModel.objects.filter(owner=user)


def user_can_access_post(user, post):
	return user.is_superuser or post.owner_id == user.id


def decrypt_post_text_fields(post, decryption_key=None):
	if is_henon_key(post.aes_key):
		post.Nama = henon_decrypt_text(post.Nama, post.aes_key)
		post.Password = henon_decrypt_text(post.Password, post.aes_key)
		post.Alamat = henon_decrypt_text(post.Alamat, post.aes_key)
		post.NIK = henon_decrypt_text(post.NIK, post.aes_key)
		return

	aes_key = unwrap_aes_key(post.aes_key, decryption_key, post.key_salt or None)
	post.Nama = aes_decrypt_text(post.Nama, aes_key)
	post.Password = aes_decrypt_text(post.Password, aes_key)
	post.Alamat = aes_decrypt_text(post.Alamat, aes_key)
	post.NIK = aes_decrypt_text(post.NIK, aes_key)


def decrypt_post_image_bytes(post, encrypted_payload, decryption_key=None):
	if is_henon_key(post.aes_key):
		return henon_decrypt_image_bytes(encrypted_payload, post.aes_key)

	aes_key = unwrap_aes_key(post.aes_key, decryption_key, post.key_salt or None)
	return aes_decrypt_image_bytes(encrypted_payload, aes_key)


def ciphertext_preview_png(encrypted_payload):
	try:
		ciphertext = get_payload_ciphertext_bytes(encrypted_payload)
	except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError):
		ciphertext = encrypted_payload

	values = np.frombuffer(ciphertext, dtype=np.uint8)
	if values.size == 0:
		values = np.zeros(1, dtype=np.uint8)

	side = int(np.ceil(np.sqrt(values.size)))
	padded = np.pad(values, (0, side * side - values.size), mode='constant')
	noise = padded.reshape((side, side))
	noise = cv2.resize(noise, (256, 256), interpolation=cv2.INTER_NEAREST)
	ok, buffer = cv2.imencode('.png', noise)
	if not ok:
		raise Http404('Ciphertext tidak bisa divisualisasikan.')
	return BytesIO(buffer).getvalue()

def get_secured_image(img, action, a, b, d):
    #---------------Read Image to Encrypt---------------
    Mod = 256
    a=int(a)
    b=int(b)
    d=int(d)
    rows, cols, ch = img.shape
    q = np.zeros([rows, cols, ch])
    key = np.array([[1,a],[b,a*b+1]])
    keyinvers=np.linalg.inv(key)
    keyinvers = keyinvers.astype(int)
    i=0
    if action == 'ENKRIPSI':
        if cols % 2 == 1:
            img = cv2.copyMakeBorder(img, 0, 0, 0, 1, cv2.BORDER_REPLICATE)
            rows, cols, ch = img.shape
            q = np.zeros([rows, cols, ch])
        # Enkripsi Hill Cipher
        for x in range (0, rows):
            for y in range (0, cols, 2):
                m=img[x,y:y+2,:]
                n=(np.matmul(key,m % Mod)) % Mod
                q[x,y:y+2,:]=n
        img=q.astype(np.uint8)
        #Enkripsi Arnold Cat Map
        while i<d:
            rows, cols, ch = img.shape
            if (rows == cols):
                n = rows
                img2 = np.zeros([rows, cols, ch])
                for x in range(0, rows):
                    for y in range(0, cols):
                        k=[x,y]
                        l=np.matmul(key,k)%n
                        img2[x,y] = img[l[0],l[1]]
                img=img2
            i=i+1
        encrypted=img.astype(np.uint8)
        return encrypted
    elif action == 'DEKRIPSI':
        #Dekripsi Arnold Cat Map
        while i<d:
            rows, cols, ch = img.shape
            if (rows == cols):
                n = rows
                img2 = np.zeros([rows, cols, ch])
                for x in range(0, rows):
                    for y in range(0, cols):
                        k=[x,y]
                        l=np.matmul(keyinvers,k)%n
                        img2[x,y] = img[l[0],l[1]]
                img=img2
            i=i+1
        img2=img.astype(np.uint8)
        #Dekripsi Hill Cipher
        rows, cols, ch = img.shape
        p = np.zeros([rows, cols, ch])
        for x in range (0, rows):
            for y in range (0, cols, 2):
                m=img[x,y:y+2,:]
                n=(np.matmul(keyinvers,m % Mod)) % Mod
                p[x,y:y+2,:]=n
        decrypted=p.astype(np.uint8)
        return decrypted


def get_secured_data(p):
	if len(p)%2==1:
		p=p+' '
	K = Matrix(([2, 1], [5, 3]))
	Km = Matrix(([1, 2], [3, 7]))

	c = ''
	cm = ''
	i = 0
	while i < len(p):
		P = Matrix((ord(p[i])-32, ord(p[i + 1])-32))
		C = Km*(K * P)
		cm = cm + chr((C[0] % 97)+32) + chr((C[1] % 97)+32)
		i = i + 2
	return cm

def get_data(p):
	if len(p)%2==1:
		p=p+' '
	K = Matrix(([2, 1], [5, 3]))
	Km = Matrix(([1, 2], [3, 7]))

	c = ''
	cm = ''
	i = 0
	while i < len(p):
		P = Matrix((ord(p[i])-32, ord(p[i + 1])-32))
		C = K.inv()*(Km.inv() * P)
		cm = cm + chr((C[0] % 97)+32) + chr((C[1] % 97)+32)
		i = i + 2
	return cm

@login_required(login_url='/login/')
def data(request):
	if request.user.is_superuser and request.GET.get('latency_samples'):
		try:
			sample_size = int(request.GET.get('latency_samples', '10'))
		except ValueError:
			sample_size = 10
		sample_size = max(1, min(sample_size, 100))
		results = run_wrapping_benchmark(sample_size, request.user)
		payload = json.dumps(results, indent=2)
		return HttpResponse(
			f'<pre>{payload}</pre>',
			content_type='text/html; charset=utf-8',
		)

	start_time = perf_counter()
	posts = list(visible_posts_for(request.user))
	db_latency_ms = (perf_counter() - start_time) * 1000
	decryption_form = DecryptionKeyForm(request.POST or None)
	decryption_key = request.session.get('decryption_key')
	decryption_error = ''

	if request.method == 'POST':
		if decryption_form.is_valid():
			decryption_key = decryption_form.cleaned_data['decryption_key']
			request.session['decryption_key'] = decryption_key
		else:
			decryption_key = None

	for post in posts:
		post.image=get_decrypted_image_url(post.image)
		if post.aes_key and is_henon_key(post.aes_key):
			try:
				decrypt_post_text_fields(post)
			except Exception:
				decryption_error = 'Sebagian data Henon tidak bisa didekripsi.'
				post.Nama='[gagal dekripsi]'
				post.Alamat='[gagal dekripsi]'
				post.NIK='[gagal dekripsi]'
		elif post.aes_key and post.key_salt and decryption_key:
			try:
				decrypt_post_text_fields(post, decryption_key)
			except Exception:
				decryption_error = 'Kunci dekripsi salah untuk sebagian data.'
				post.Nama='[kunci dekripsi salah]'
				post.Alamat='[kunci dekripsi salah]'
				post.NIK='[kunci dekripsi salah]'
		elif post.aes_key and not post.key_salt:
			try:
				decrypt_post_text_fields(post)
			except Exception:
				decryption_error = 'Sebagian data lama tidak bisa didekripsi.'
				post.Nama='[gagal dekripsi]'
				post.Alamat='[gagal dekripsi]'
				post.NIK='[gagal dekripsi]'
		elif post.aes_key:
			post.Nama='[masukkan kunci dekripsi]'
			post.Alamat='[masukkan kunci dekripsi]'
			post.NIK='[masukkan kunci dekripsi]'
		else:
			post.Nama=get_data(post.Nama)
			post.Alamat=get_data(post.Alamat)
			post.NIK=get_data(post.NIK)
	context = {
		'page_title':'Data anda akan tersimpan dengan aman',
		'posts':posts,
		'db_latency_ms':db_latency_ms,
		'show_ciphertext':False,
		'decryption_form':decryption_form,
		'decryption_error':decryption_error,
	}

	return render(request,'main/home.html',context)

@login_required(login_url='/login/')
def home(request):
	start_time = perf_counter()
	posts = list(visible_posts_for(request.user))
	db_latency_ms = (perf_counter() - start_time) * 1000
	for post in posts:
		post.image=get_image_url(post.image)
		post.Nama=get_payload_ciphertext_text(post.Nama)
		post.Alamat=get_payload_ciphertext_text(post.Alamat)
		post.NIK=get_payload_ciphertext_text(post.NIK)
		post.image_ciphertext=get_payload_ciphertext_text(post.image_ciphertext)
	context = {
		'page_title':'Data anda akan tersimpan dengan aman',
		'posts':posts,
		'db_latency_ms':db_latency_ms,
		'show_ciphertext':True,
	}
	return render(request,'main/home.html',context)


@login_required(login_url='/login/')
@user_passes_test(lambda user: user.is_superuser, login_url='/login/')
def benchmark(request):
	try:
		sample_size = int(request.GET.get('samples', '10'))
	except ValueError:
		sample_size = 10
	sample_size = max(1, min(sample_size, 100))
	results = run_wrapping_benchmark(sample_size, request.user)
	return JsonResponse(results, json_dumps_params={'indent': 2})


@login_required(login_url='/login/')
@user_passes_test(lambda user: user.is_superuser, login_url='/login/')
def image_benchmark(request):
	try:
		runs = int(request.GET.get('runs', '3'))
	except ValueError:
		runs = 3
	results = run_image_encryption_benchmark(runs)
	return JsonResponse(results, json_dumps_params={'indent': 2})

@login_required(login_url='/login/')
def create(request):
	post_form = PostForm()

	if request.method == 'POST':
		post_form = PostForm(request.POST, request.FILES)
		if post_form.is_valid():
			uploaded_image = post_form.cleaned_data['image']
			uploaded_image_bytes = uploaded_image.read()
			image_bytes = np.frombuffer(uploaded_image_bytes, np.uint8)
			img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

			if img is None:
				post_form.add_error('image', 'File gambar tidak bisa dibaca.')
			else:
				henon_key = generate_henon_key()
				secured_image_name = f'{uuid4().hex}.henon'
				encrypted_image = henon_encrypt_image_bytes(uploaded_image_bytes, henon_key)

				ENCRYPTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
				(ENCRYPTED_IMAGE_DIR / secured_image_name).write_bytes(encrypted_image)

				PostModel.objects.create(
						owner		= request.user,
						Nama 		= henon_encrypt_text(post_form.cleaned_data['nama'], henon_key),
						Password	= henon_encrypt_text(post_form.cleaned_data['password'], henon_key),
						NIK		= henon_encrypt_text(post_form.cleaned_data['nik'], henon_key),
						image 		= secured_image_name,
						image_ciphertext = encrypted_image.decode('utf-8'),
						Alamat		= henon_encrypt_text(post_form.cleaned_data['alamat'], henon_key),
						aes_key		= henon_key,

					)

				return HttpResponseRedirect('/data/')


	context = {
		'page_title':'Pendaftaran',
		'post_form':post_form
	}

	return render(request,'main/create.html',context)


@login_required(login_url='/login/')
def encrypted_image(request, filename):
	requested_name = Path(filename).name
	existing_post = PostModel.objects.filter(image=requested_name).first()
	if existing_post and not user_can_access_post(request.user, existing_post):
		raise Http404('Gambar tidak ditemukan.')

	image_path = find_stored_image(filename)
	if image_path:
		if image_path.suffix == '.henon':
			return HttpResponse(
				get_payload_ciphertext_bytes(image_path.read_bytes()),
				content_type='image/png',
			)
		if image_path.suffix == '.aes':
			return HttpResponse(
				ciphertext_preview_png(image_path.read_bytes()),
				content_type='image/png',
			)
		return FileResponse(open(image_path, 'rb'), content_type='image/png')

	post = visible_posts_for(request.user).filter(image=requested_name).first()
	if post and post.image_ciphertext:
		if is_henon_key(post.aes_key):
			return HttpResponse(
				get_payload_ciphertext_bytes(post.image_ciphertext),
				content_type='image/png',
			)
		return HttpResponse(
			ciphertext_preview_png(post.image_ciphertext.encode('utf-8')),
			content_type='image/png',
		)

	raise Http404('Gambar tidak ditemukan.')


@login_required(login_url='/login/')
def decrypted_image(request, filename):
	image_path = find_stored_image(filename)
	post = visible_posts_for(request.user).filter(image=Path(filename).name).first()
	if post and post.aes_key:
		decryption_key = request.session.get('decryption_key')
		if post.key_salt and not decryption_key and not is_henon_key(post.aes_key):
			raise Http404('Masukkan kunci dekripsi terlebih dahulu.')
		try:
			encrypted_payload = (
				image_path.read_bytes()
				if image_path
				else post.image_ciphertext.encode('utf-8')
			)
			image_bytes = decrypt_post_image_bytes(
				post,
				encrypted_payload,
				decryption_key if post.key_salt and not is_henon_key(post.aes_key) else None,
			)
		except (ValueError, KeyError):
			raise Http404('Gambar tidak bisa didekripsi.')
		image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
		if image is None:
			raise Http404('Gambar tidak bisa dibaca.')
		ok, buffer = cv2.imencode('.png', image)
		if not ok:
			raise Http404('Gambar tidak bisa ditampilkan.')
		return HttpResponse(BytesIO(buffer).getvalue(), content_type='image/png')

	if not image_path:
		raise Http404('Gambar tidak ditemukan.')

	encrypted = cv2.imread(str(image_path))
	if encrypted is None:
		raise Http404('Gambar tidak bisa dibaca.')

	decrypted = get_secured_image(encrypted, 'DEKRIPSI', 2, 3, 2)
	ok, buffer = cv2.imencode('.png', decrypted)
	if not ok:
		raise Http404('Gambar tidak bisa didekripsi.')

	return HttpResponse(BytesIO(buffer).getvalue(), content_type='image/png')


def login(request):
	login_form = LoginForm()
	context = {
		'page_title': 'Login',
		'login_form':login_form
	}
	user=None
	if request.method == 'GET':
		if request.user.is_authenticated == True:
			return redirect('/data')
		else:
			return render(request, 'main/login.html', context)
	elif request.method == "POST":
		username_login = request.POST['nama']
		password_login = request.POST['password']

		user = authenticate(request, username=username_login, password=password_login)
		print(user)
		if user is not None:
			auth_login(request, user)
			return redirect('/data')
		else:
			print(username_login)
			print(password_login)
			print('Username atau password anda salah, silahkan masukkan dengan benar!')
			return redirect('/login')
	return render(request, 'main/login.html', context)



def logout(request):
	auth_logout(request)
	return redirect('/')
