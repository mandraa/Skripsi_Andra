from flask import Flask, render_template, request, redirect, url_for, Response, flash, session
import threading
import mysql.connector
from datetime import datetime
import os
import time
import cv2
import sys
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import secrets

# # Import fungsi dari combine_detect.py
# # Kita akan memodifikasi ini sedikit untuk memungkinkan mengakses webcam dari 2 tempat
# from combine_detect import run_combined_detection, parse_opt, main as detection_main

app = Flask(__name__, template_folder='templates')

app.secret_key = secrets.token_hex(32)  # Gunakan kunci rahasia yang aman

# Koneksi database - sesuaikan dengan konfigurasi Anda
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "skripsi",
    "port": 3306
}

# # Variabel global untuk menyimpan frame dari webcam untuk stream web
# global_frame = None
# detection_thread = None
# detection_running = False

def get_db_connection():
    """Membuat koneksi ke database"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error koneksi database: {e}")
        return None
    
def login_required(f):
    """Decorator untuk memastikan pengguna sudah login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Anda harus login terlebih dahulu!', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Halaman login untuk admin"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM admin WHERE username = %s", (username,))
            user = cursor.fetchone()
            
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                
                flash(f'Selamat Datang, {user["username"]}!',  'success')
                return redirect(url_for('beranda')) # Redirect ke halaman utama setelah login berhasil
            
            else:
                flash('Username atau password salah!', 'danger')

            cursor.close()
            conn.close()
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Halaman registrasi untuk admin baru"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validasi
        if password != confirm_password:
            flash('Password dan konfirmasi password tidak cocok!', 'danger')
            return render_template('register.html')
        
        if len(password) < 8:
            flash('Password harus minimal 8 karakter!', 'danger')
            return render_template('register.html')
        
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            
            # Cek apakah username atau email sudah ada
            cursor.execute("SELECT * FROM admin WHERE username = %s OR email = %s", (username, email))
            existing_user = cursor.fetchone()
            
            if existing_user:
                flash('Username atau email sudah terdaftar!', 'danger')
            else:
                # Hash password sebelum disimpan
                password_hash = generate_password_hash(password)
                
                # Simpan data admin baru
                cursor.execute("INSERT INTO admin (username, email, password) VALUES (%s, %s, %s)", 
                               (username, email, password_hash))
                conn.commit()
                
                flash('Registrasi berhasil! Silakan login.', 'success')
                return redirect(url_for('login'))
            # Tutup koneksi
            cursor.close()
            conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout admin"""
    session.clear()
    flash('Anda telah logout.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def beranda():
    """Halaman utama dengan data pelanggaran"""
    conn = get_db_connection()
    violations = []
    
    if conn:
        cursor = conn.cursor(dictionary=True)
        # Ambil data pelanggaran dari database, urutkan dari yang terbaru
        cursor.execute("""
            SELECT * FROM safety_violations 
            ORDER BY violation_id ASC
        """)
        violations = cursor.fetchall()
        cursor.close()
        conn.close()
    
    # Hitung total pelanggaran per orang
    violation_counts = {}
    for violation in violations:
        person_names = violation['person_name'].split(', ')
        for person in person_names:
            if person in violation_counts:
                violation_counts[person] += 1
            else:
                violation_counts[person] = 1
    
    # Konversi ke format yang mudah digunakan di template
    violation_counts = [{"name": name, "count": count} for name, count in violation_counts.items()]
    # Urutkan berdasarkan jumlah pelanggaran (dari tinggi ke rendah)
    violation_counts.sort(key=lambda x: x["count"], reverse=True)
    
    return render_template('index.html', 
                          violations=violations, 
                          violation_counts=violation_counts)

@app.route('/view_image/<path:image_path>')
@login_required
def view_image(image_path):
    """Tampilkan gambar pelanggaran"""
    # Jika path berisi 'violation_screenshots/', kita menggunakan path relatif dari root project
    if 'violation_screenshots/' in image_path:
        # Ekstrak bagian path setelah 'violation_screenshots/'
        rel_path = image_path.split('violation_screenshots/')[-1]
        # Buat path lengkap ke file
        full_path = os.path.join('violation_screenshots', rel_path)
    else:
        full_path = image_path
        
    # Periksa apakah file ada
    if not os.path.exists(full_path):
        return "Gambar tidak ditemukan", 404
        
    # Baca gambar dan tampilkan
    with open(full_path, 'rb') as f:
        image_data = f.read()
    
    return Response(image_data, mimetype='image/jpeg')

@app.route('/daily_report')
@login_required
def daily_report():
    """Halaman laporan harian"""
    conn = get_db_connection()
    daily_data = []
    
    if conn:
        cursor = conn.cursor(dictionary=True)
        # Ambil data pelanggaran dikelompokkan per hari
        cursor.execute("""
            SELECT date_day, COUNT(*) as total_violations
            FROM safety_violations 
            GROUP BY date_day
            ORDER BY STR_TO_DATE(SUBSTRING_INDEX(date_day, ',', -1), ' %d-%m-%Y') ASC
        """)
        daily_data = cursor.fetchall()
        cursor.close()
        conn.close()
    
    return render_template('daily_report.html', daily_data=daily_data)

@app.route('/person_report/<person_name>')
@login_required
def person_report(person_name):
    """Halaman laporan per orang"""
    conn = get_db_connection()
    person_violations = []
    
    if conn:
        cursor = conn.cursor(dictionary=True)
        # Ambil data pelanggaran untuk orang tertentu
        cursor.execute("""
            SELECT * FROM safety_violations 
            WHERE person_name LIKE %s
            ORDER BY violation_id ASC
        """, (f"%{person_name}%",))
        person_violations = cursor.fetchall()
        cursor.close()
        conn.close()
        
    # Tentukan URL foto profil
    # Asumsikan foto profil ada di folder face_dataset/nama_person dengan format nama_person1.jpg 
    profile_image_path = f"face_dataset/{person_name.replace(' ', '_')}/{person_name.replace(' ', '_')}1.jpg"
    
    # Periksa apakah file foto profil ada
    if os.path.exists(profile_image_path):
        profile_url = url_for('view_image', image_path=profile_image_path)
    else:
        # Jika tidak ada foto profil, gunakan foto default atau kosong
        profile_url = url_for('static', filename='images/default-profile.png')  # atau bisa kosong ""
    
    return render_template('person_report.html', 
                          person_name=person_name,
                          violations=person_violations,
                          url=profile_url)

if __name__ == '__main__':
    # Impor numpy di sini untuk menghindari masalah dengan threading
    import numpy as np
    
    # Jalankan server Flask
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)