import argparse
import cv2
import numpy as np
import os
import platform
import pathlib
import sys
# Patch untuk Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import time
from datetime import datetime
from pathlib import Path
import mtcnn
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
import pickle
import winsound  # For Windows sound playback
import mysql.connector
from mysql.connector import Error

# Add YOLOv5 to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Assume YOLOv5 is in the same directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # Add ROOT to PATH

# Import YOLOv5 modules
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    LOGGER, check_img_size, check_requirements, colorstr, cv2,
    increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh
)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device

# --- Face Recognition Configuration ---
# Initialize MTCNN for face detection
face_detector = mtcnn.MTCNN()

# Initialize FaceNet model
facenet_model = None  # Will load later
l2_normalizer = Normalizer('l2')

# Configuration parameters
detection_confidence_threshold = 0.95
required_size = (160, 160)  # FaceNet requires 160x160 input
similarity_threshold = 0.7  # Cosine similarity threshold for face recognition

# --- Global Variables for Violation Tracking ---
detected_violators = {}  # Dictionary untuk menyimpan pelanggar dan waktu terakhir terdeteksi
violation_cooldown = 60  # Waktu cooldown dalam detik sebelum pelanggar yang sama bisa dicatat lagi
recorded_violations = set()  # Set untuk melacak pelanggaran yang sudah dicatat (agar hanya sekali)

def create_database_from_folder(dataset_path="face_dataset", database_path="face_recognition_database.pkl"):
    # Initialize database
    database = {
        'names': [],
        'embeddings': []
    }
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist!")
        return False
    
    # Get list of persons (folders)
    persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not persons:
        print(f"Error: No person folders found in {dataset_path}!")
        return False
    
    print(f"Found {len(persons)} persons: {', '.join(persons)}")
    
    # Process each person
    total_faces = 0
    
    for person in persons:
        person_dir = os.path.join(dataset_path, person)
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"Warning: No images found for {person}!")
            continue
        
        print(f"Processing {len(image_files)} images for {person}...")
        
        # Process each image
        person_faces = 0
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}!")
                continue
            
            # Detect faces
            face_boxes, _ = detect_face(img)
            if not face_boxes:
                print(f"Warning: No face detected in {img_path}!")
                continue
            
            # Normalize faces
            face_imgs, _ = normalize_faces(img, face_boxes)
            if not face_imgs:
                print(f"Warning: Failed to process face in {img_path}!")
                continue
            
            # Get embeddings
            embeddings = get_embeddings(face_imgs)
            
            # Add to database
            for embedding in embeddings:
                database['names'].append(person)
                database['embeddings'].append(embedding)
                person_faces += 1
            
        total_faces += person_faces
        print(f"Added {person_faces} face(s) for {person}")
    
    # Save database
    with open(database_path, 'wb') as file:
        pickle.dump(database, file)
    
    print(f"\nDatabase creation complete!")
    print(f"Added {total_faces} face(s) from {len(persons)} person(s)")
    print(f"Database saved to {database_path}")
    return True

# --- Face Recognition Functions ---
def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def detect_face(img):
    faces = face_detector.detect_faces(img)
    face_boxes = []
    face_scores = []
    for face in faces:
        confidence = face['confidence']
        if confidence >= detection_confidence_threshold:
            x, y, w, h = face['box']
            face_boxes.append([x, y, w, h])
            face_scores.append(confidence)
    return face_boxes, face_scores

def normalize_faces(img, face_boxes):
    face_imgs = []
    face_locations = []
    for box in face_boxes:
        face_img, start_point, end_point = get_face(img, box)
        if face_img.size != 0:
            face_img = cv2.resize(face_img, required_size)
            face_imgs.append(face_img)
            face_locations.append((start_point, end_point))
    return face_imgs, face_locations

def get_embeddings(face_imgs):
    face_imgs = [cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) for face_img in face_imgs]
    face_imgs = np.array([img.astype('float32') for img in face_imgs])
    face_imgs = (face_imgs - 127.5) / 128.0
    embeddings = facenet_model.predict(face_imgs)
    embeddings = l2_normalizer.transform(embeddings)
    return embeddings

def load_database(database_path):
    with open(database_path, 'rb') as file:
        database = pickle.load(file)
    return database

def recognize_face_cosine_similarity(embedding, database, threshold=similarity_threshold):
    if len(database['embeddings']) == 0:
        return "Unknown", 0.0
    similarities = []
    for ref_embedding in database['embeddings']:
        similarity = 1 - cosine(embedding, ref_embedding)
        similarities.append(similarity)
    idx = np.argmax(similarities)
    max_similarity = similarities[idx]
    if max_similarity >= threshold:
        return database['names'][idx], max_similarity
    else:
        return "Unknown", max_similarity

def recognize_faces(img, database):
    face_boxes, face_scores = detect_face(img)
    if len(face_boxes) == 0:
        return [], [], []
    face_imgs, face_locations = normalize_faces(img, face_boxes)
    if len(face_imgs) == 0:
        return [], [], []
    embeddings = get_embeddings(face_imgs)
    predictions = []
    similarities = []
    for embedding in embeddings:
        name, similarity = recognize_face_cosine_similarity(embedding, database)
        predictions.append(name)
        similarities.append(similarity)
    return predictions, similarities, face_locations

def create_db_connection(host_name, user_name, user_password, db_name, port):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            password=user_password,
            database=db_name,
            port=port  # Use the port parameter passed to the function
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection

def save_violation_to_db(connection, violation_type, number_of_persons, violator_name=None, image_path=None):
    global detected_violators, violation_cooldown
    cursor = None
    try:
        # Filter nama-nama unknown dari violator_name
        if violator_name:
            violator_name = [name for name in violator_name if name.lower() != "unknown"]
            
        # Jika tidak ada nama yang valid setelah filter, lewati penyimpanan
        if not violator_name or len(violator_name) == 0:
            print("Semua pelanggar tidak dikenali (unknown). Melewati penyimpanan ke database.")
            return
        
        # Buat kunci unik berdasarkan nama pelanggar (diurutkan)
        violation_key = "_".join(sorted(violator_name))
        current_time = time.time()
        
        # Pengecekan apakah pelanggaran ini sudah pernah dicatat
        if violation_key in detected_violators:
            last_detection_time = detected_violators[violation_key]
            time_elapsed = current_time - last_detection_time
            
            if time_elapsed < violation_cooldown:
                time_remaining = violation_cooldown - time_elapsed
                print(f"Pelanggaran untuk {violation_key} masih dalam masa cooldown. {time_remaining:.1f} detik lagi sebelum dicatat ulang.")
                return
            else:
                print(f"Cooldown untuk {violation_key} sudah selesai. Mencatat pelanggaran baru.")
        
        # Update waktu terakhir deteksi untuk pelanggar
        detected_violators[violation_key] = current_time
        
        if connection.is_connected():
            cursor = connection.cursor()
            person_name_str = ", ".join (violator_name) 
            now = datetime.now()
            hari_names = {
                0: "Senin",
                1: "Selasa",
                2: "Rabu",
                3: "Kamis",
                4: "Jumat",
                5: "Sabtu",
                6: "Minggu"
            }
            hari_index = now.weekday()
            hari_name = hari_names[hari_index]
            
            # Format tanggal: Hari, DD-MM-YYYY
            date_day = f"{hari_name}, {now.strftime('%d-%m-%Y')}"
            
            # Format waktu: HH:MM:SS
            time_only = now.strftime("%H:%M:%S")
            
            # Insert data into the database
            query = """INSERT INTO safety_violations 
                     (violation_type, number_of_persons, person_name, image_path, date_day, time_only) 
                     VALUES (%s, %s, %s, %s, %s, %s)"""
            values = (violation_type, number_of_persons, person_name_str, image_path, date_day, time_only)
            cursor.execute(query, values)
            connection.commit()
            print("Violation record for inserted successfully")
    except Error as err:
        print(f"Error: '{err}'")
    finally:
        if cursor is not None:
            cursor.close()

def is_working_hours(work_hours_config):
    """
    Memeriksa apakah waktu saat ini adalah jam kerja.
    Return True jika jam kerja, False jika jam istirahat atau di luar jam kerja.
    
    :param work_hours_config: Dictionary berisi konfigurasi jam kerja
    :return: Boolean
    """
    now = datetime.now()
    current_time = now.time()
    current_day = now.weekday()  # 0 = Senin, 6 = Minggu
    
    # Cek apakah hari kerja (Default: Senin-Jumat)
    working_days = work_hours_config.get('working_days', [0, 1, 2, 3, 4, 5, 6])
    if current_day not in working_days:
        return False, "Hari Libur"
    
    # Ambil konfigurasi waktu kerja
    work_start_morning = work_hours_config.get('work_start_morning')
    work_end_morning = work_hours_config.get('work_end_morning')
    lunch_break_end = work_hours_config.get('lunch_break_end')
    work_end_afternoon = work_hours_config.get('work_end_afternoon')
    
    # Cek apakah waktu saat ini adalah jam kerja
    is_morning_shift = work_start_morning <= current_time < work_end_morning
    is_afternoon_shift = lunch_break_end <= current_time < work_end_afternoon
    
    # Cek status saat ini
    if is_morning_shift:
        return True, "Shift Pagi"
    elif work_end_morning <= current_time < lunch_break_end:
        return False, "Istirahat Siang"
    elif is_afternoon_shift:
        return True, "Shift Siang"
    elif current_time < work_start_morning:
        return False, "Belum Jam Kerja"
    else:
        return False, "Jam Kerja Telah selesai"
    
            
def save_screenshot(frame, is_violation, person_name="unknown"):
    # Create directory if it doesn't exist
    screenshots_dir = "violation_screenshots"
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    status = "violation" if is_violation else "safe"
    safe_name = ''.join(c for c in person_name if c.isalnum() or c in [' ', '_']).replace(' ', '_')
    filename = f"{screenshots_dir}/{status}_{safe_name}_{timestamp}.jpg"
    
    # Save the image
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved to {filename}")
    
    return filename

# --- Combined Detection Run Function ---
def run_combined_detection(
    weights=ROOT / "best2.pt",  # YOLOv5 model path
    source="1",  # Webcam source (0 is default)
    database_path="face_recognition_database.pkl",  # Face recognition database
    imgsz=(640, 640),  # YOLOv5 inference size
    conf_thres=0.5,  # YOLOv5 confidence threshold
    iou_thres=0.45,  # YOLOv5 NMS IOU threshold
    max_det=1000,  # Maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    face_sim_threshold=0.7,  # Face recognition similarity threshold
    db_host="localhost",  # Database host
    db_user="root",  # Database user
    db_password="",  # Database password
    db_name="skripsi",  # Database name
    db_port=3306,  # Database port
    check_working_hours=True,  # Enable working hours check
    work_hours_config=None  # Working hours configuration
):
    global facenet_model
    
    # Database connection
    print("Connecting to MySQL database...")
    db_connection = create_db_connection(db_host, db_user, db_password, db_name, db_port)

    # Load FaceNet model
    print("Loading FaceNet model...")
    facenet_model = load_model('facenet_keras.h5')
    
    # Check if database exists, if not create it
    if not os.path.exists(database_path):
        print(f"Database {database_path} does not exist, creating from face_dataset folder...")
        if not create_database_from_folder(dataset_path="face_dataset", database_path=database_path):
            print("Error creating database. Exiting.")
            return
    
    # Load face recognition database
    print("Loading face recognition database...")
    try:
        face_database = load_database(database_path)
        print(f"Loaded database with {len(face_database['names'])} persons")
    except Exception as e:
        print(f"Error loading face database: {e}")
        return

    # Load YOLOv5 model
    print("Loading YOLOv5 model...")
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=None)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # Check image size

    # Dataloader
    print("Starting video capture...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # Warmup
    
    print("Starting combined detection. Press ESC to exit.")
    
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if check_working_hours and work_hours_config:
        system_active, status_message = is_working_hours(work_hours_config)
        if system_active:
            print(f"{current_datetime} - Sistem aktif: {status_message}")
        else:
            print(f"{current_datetime} - Sistem tidak aktif: {status_message}")
    else:
        print(f"{current_datetime} - Sistem aktif: Mode pengecekan jam kerja tidak aktif")
    
    last_status_check_time = time.time()
    system_active = True
    current_status = "Sistem Aktif"
    status_check_interval = 10  # Cek status setiap 60 detik
    previous_status_message = ""
    
    while True:
        # Cek status jam kerja setiap interval tertentu
        current_time = time.time()
        if check_working_hours and work_hours_config and (current_time - last_status_check_time > status_check_interval):
            system_active, status_message = is_working_hours(work_hours_config)
            current_status = f"Status: {status_message}"
            last_status_check_time = current_time
            
             # Cek apakah status berubah menjadi "Jam Kerja Telah selesai"
            if status_message == "Jam Kerja Telah selesai" and previous_status_message != "Jam Kerja Telah selesai":
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {status_message}")
                # Tampilkan pesan penutupan
                for countdown in range(5, 0, -1):
                    ret, frame = cap.read()
                    if ret:
                        # Tambahkan pesan penutupan
                        cv2.putText(frame, f"SISTEM TIDAK AKTIF - {current_status}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"Sistem akan dimatikan dalam {countdown} detik", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(frame, current_datetime, (10, frame.shape[0] - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imshow("Combined Detection System", frame)
                        cv2.waitKey(1000)  # Delay 1 detik
                
                # Keluar dari loop
                break
            
            previous_status_message = status_message
            
            if system_active:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Sistem aktif: {status_message}")
            else:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Sistem tidak aktif: {status_message}")
                
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        
        # Tampilkan status sistem pada frame
        if not system_active and check_working_hours:
            # Jika di luar jam kerja, tampilkan pesan dan lanjutkan loop
            # Tambahkan tanggal dan waktu pada tampilan
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"SISTEM TIDAK AKTIF - {current_status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, current_datetime, (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Combined Detection System", frame)
            
            # Exit on ESC
            if cv2.waitKey(1) == 27:
                break
                
            # Delay before next check to reduce CPU usage
            time.sleep(3)
            continue
        
        # Tambahkan status dan waktu pada tampilan
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"SISTEM AKTIF - {current_status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Make a copy of the frame for face recognition
        face_frame = frame.copy()
        
        # --- YOLOv5 Detection ---
        # Prepare image for YOLOv5
        img = frame.copy()
        img = img[..., ::-1]  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.permute(2, 0, 1).float()  # HWC to CHW
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # Expand for batch dim
        
        # YOLOv5 inference
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
        
        # Variables for helmet detection
        helmet_count = 0
        head_count = 0
        
        # Process YOLOv5 predictions
        for i, det in enumerate(pred):
            annotator = Annotator(frame, line_width=3, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                
                # Process detections
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # Integer class
                    label = names[c] if False else f"{names[c]} {conf:.2f}"
                    
                    # Count helmets and heads
                    if names[c] == "helmet":
                        helmet_count += 1
                    elif names[c] == "head" or names[c] == "person":
                        head_count += 1
                    
                    # Draw bounding box
                    annotator.box_label(xyxy, label, color=colors(c, True))
        
        
            # --- Face Recognition ---
            predictions, similarities, face_locations = recognize_faces(face_frame, face_database)
        
        
        # Draw face recognition results
        for name, similarity, loc in zip(predictions, similarities, face_locations):
            start_point, end_point = loc
            color = (0, 255, 0) if similarity >= face_sim_threshold else (0, 0, 255)
            label = f"{name}: {similarity:.2f}" if similarity >= face_sim_threshold else f"Unknown: {similarity:.2f}"
            cv2.rectangle(frame, start_point, end_point, color, 2)
            cv2.putText(frame, label, (start_point[0], start_point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # --- Check for Violations ---
        # Hitung jumlah orang berdasarkan deteksi wajah dan deteksi helm/kepala
        face_count = len(face_locations)
        total_people = max(head_count, helmet_count, face_count)  # Jumlah orang adalah nilai maksimum

        # Siapkan daftar untuk menyimpan nama orang yang terdeteksi
        detected_person_names = []

        # Jika jumlah deteksi wajah lebih sedikit dari total orang yang terdeteksi,
        # tambahkan "unknown" untuk orang-orang yang tidak dikenali wajahnya
        if face_count < total_people:
            # Tambahkan semua nama yang terdeteksi
            for name in predictions:
                detected_person_names.append(name)
    
            # Tambahkan "unknown" untuk orang-orang yang tidak terdeteksi wajahnya
            for i in range(total_people - face_count):
                detected_person_names.append("unknown")
        else:
            # Jika jumlah deteksi wajah sama dengan atau lebih banyak dari total orang
            # Gunakan nama-nama dari deteksi wajah
            detected_person_names = predictions[:total_people]

        # Nama untuk screenshot (gunakan nama gabungan dengan underscore)
        if detected_person_names:
            screenshot_name = "_".join(detected_person_names)
        else:
            screenshot_name = "unknown"
            
            
        if total_people == 0:
            # Tidak ada orang yang terdeteksi
            status_text = "No people detected"
            print(status_text)
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # Kuning
            
        elif helmet_count < total_people:
            # Ada orang yang tidak memakai helm (PELANGGARAN)
            not_wearing_helmet = total_people - helmet_count
            warning_text = f"Violation: {not_wearing_helmet} of {total_people} person(s) not wearing helmet"
            print(warning_text)
            
            # Filter hanya orang yang tidak memakai helm
            violators = []
            for i, name in enumerate(detected_person_names):
                # Jika orang ini tidak terdeteksi memakai helm (indeks di luar jumlah helm)
                if i >= helmet_count and name.lower() != "unknown":
                    violators.append(name)
    
            # Jika tidak ada violator yang dikenali, lewati
            if not violators:
                print("Tidak ada pelanggar yang teridentifikasi. Melewati penyimpanan.")
            else:
                # Lakukan screenshot hanya jika ada pelanggar yang teridentifikasi
                screenshot_name = "_".join(violators)
                image_path = save_screenshot(frame, True, screenshot_name)
        
                # Coba simpan pelanggaran ke database
                if db_connection is not None and db_connection.is_connected():
                    save_violation_to_db(db_connection, "not wearing helmet", len(violators), violators, image_path)
    
            if platform.system() == "Windows":
                winsound.Beep(1000, 500)
    
            # Tambahkan teks peringatan ke frame
            cv2.putText(frame, warning_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        else:
            # Di sini helmet_count >= total_people
            safe_text = f"Safe: {helmet_count} helmets on {total_people} people"
            print(safe_text)
    
            cv2.putText(frame, safe_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
             
              
        # Tambahkan tanggal dan waktu pada tampilan
        cv2.putText(frame, current_datetime, (10, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show results
        cv2.imshow("Combined Detection System", frame)
        
        # Exit on ESC
        if cv2.waitKey(1) == 27:
            break
        
            
    # Clean up
    print("Mematikan sistem...")
    cap.release()
    cv2.destroyAllWindows()
    print("Sistem telah dimatikan.")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "best2.pt", help="YOLOv5 model path")
    parser.add_argument("--source", type=str, default="1", help="video source (1 for webcam)")
    parser.add_argument("--database", type=str, default="face_recognition_database.pkl", help="face recognition database path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--face-sim-threshold", type=float, default=0.7, help="face similarity threshold")
    parser.add_argument("--create-db", action="store_true", help="force create database")
    parser.add_argument("--db-host", type=str, default="localhost", help="MySQL database host")
    parser.add_argument("--db-user", type=str, default="root", help="MySQL database user")
    parser.add_argument("--db-password", type=str, default="", help="MySQL database password")
    parser.add_argument("--db-name", type=str, default="skripsi", help="MySQL database name")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL database port")
    parser.add_argument("--check-working-hours", action="store_true", help="Enable checking of working hours")
    parser.add_argument("--work-start", type=str, default="08:00", help="Working hours start time (HH:MM)")
    parser.add_argument("--break-start", type=str, default="12:00", help="Lunch break start time (HH:MM)")
    parser.add_argument("--break-end", type=str, default="13:00", help="Lunch break end time (HH:MM)")
    parser.add_argument("--work-end", type=str, default="23:51", help="Working hours end time (HH:MM)")
    parser.add_argument("--working-days", type=str, default="0,1,2,3,4,5,6", 
                       help="Working days (0=Monday, 6=Sunday), comma-separated")
    opt = parser.parse_args()
    return opt

def main(opt):
    print(colorstr("bold", "Combined Helmet Detection and Face Recognition System"))
    # Check requirements
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    
    # Parse working hours configuration
    work_hours_config = {
        'working_days': [int(day) for day in opt.working_days.split(',')],
        'work_start_morning': datetime.strptime(opt.work_start, "%H:%M").time(),
        'work_end_morning': datetime.strptime(opt.break_start, "%H:%M").time(),
        'lunch_break_end': datetime.strptime(opt.break_end, "%H:%M").time(),
        'work_end_afternoon': datetime.strptime(opt.work_end, "%H:%M").time()
    }
    
    print(f"Working Hours Configuration:")
    print(f"- Working Days: {', '.join(['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'][d] for d in work_hours_config['working_days'])}")
    print(f"- Morning Shift: {opt.work_start} - {opt.break_start}")
    print(f"- Lunch Break: {opt.break_start} - {opt.break_end}")
    print(f"- Afternoon Shift: {opt.break_end} - {opt.work_end}")
    
    if opt.create_db:
        print("Creating face recognition database...")
        create_database_from_folder(dataset_path=opt.dataset, database_path=opt.database)
    
    # Run combined system
    run_combined_detection(
        weights=opt.weights,
        source=opt.source,
        database_path=opt.database,
        device=opt.device,
        conf_thres=opt.conf_thres,
        face_sim_threshold=opt.face_sim_threshold,
        db_host=opt.db_host,
        db_user=opt.db_user,
        db_password=opt.db_password,
        db_name=opt.db_name,
        db_port=opt.db_port,
        check_working_hours=opt.check_working_hours,
        work_hours_config=work_hours_config
        
    )

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)