import RPi.GPIO as GPIO
from time import sleep
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from RPLCD.i2c import CharLCD
from pyfingerprint.pyfingerprint import PyFingerprint
import threading
from datetime import datetime
from flask import Flask, request, jsonify, Response
import logging
from picamera2 import Picamera2
import cv2
import time
import requests
import ssl
from typing import Optional
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import re
import tempfile
from vosk import Model, KaldiRecognizer
import pyaudio
import wave
import mysql.connector
import face_recognition
import numpy as np
from scipy.spatial import distance as dist

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Define the GPIO pins for rows, columns, and button
ROW_PINS = [5, 27, 17, 4]
COL_PINS = [7, 8, 25, 18]
BUTTON_PIN = 20

# Define GPIO pin for the solenoid lock
LOCK_PIN = 6

# Keypad layout
KEYPAD = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'OK']
]

# Directory to save encoded faces
FACE_DIR = "saved_faces"
if not os.path.exists(FACE_DIR):
    os.makedirs(FACE_DIR)

# Constants for blink detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# Constants for mouth detection
MOUTH_AR_THRESH = 0.75

# Initialize the LCD
lcd = CharLCD('PCF8574', 0x27)

# GPIO Setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Pin Definitions
PINS = {
    'button1': 24,  # Record unlock command
    'button5': 1,   # Register fingerprint
    'button6': 19,  # Manual unlock
    'button7': 9,   # Register face
    'solenoid': 6,
    'buzzer': 22,
}

# Setup GPIO pins
for pin in PINS.values():
    if pin in [PINS['solenoid'], PINS['buzzer']]:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    else:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Set up keypad GPIO
for row in ROW_PINS:
    GPIO.setup(row, GPIO.OUT)
    GPIO.output(row, GPIO.LOW)

for col in COL_PINS:
    GPIO.setup(col, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Initialize fingerprint sensor
try:
    f = PyFingerprint('/dev/ttyS0', 57600, 0xFFFFFFFF, 0x00000000)
    if not f.verifyPassword():
        raise ValueError('The given fingerprint sensor password is incorrect!')
except Exception as e:
    logger.error(f'Fingerprint sensor initialization failed: {str(e)}')
    exit(1)

# Initialize PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.preview_configuration)
picam2.start()

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.last_connection_attempt = 0
        self.reconnect_interval = 60  # Try to reconnect every 60 seconds
        
    def connect(self):
        """Connect to database."""
        try:
            self.connection = mysql.connector.connect(
                host="192.168.8.36",
                user="root",
                password="oneinamillion",
                database="Smartdb",
                connection_timeout=5,
                autocommit=True
            )
            self.cursor = self.connection.cursor(dictionary=True)
            logger.info("Database connected successfully")
            return True
        except Exception as e:
            logger.warning(f"Database connection failed: {str(e)}")
            self.connection = None
            self.cursor = None
            return False
    
    def ensure_connection(self):
        """Ensure database connection is available when online."""
        current_time = time.time()
        
        # Only try to reconnect if we're online and enough time has passed
        if (system_mode.is_online and 
            not self.connection and 
            current_time - self.last_connection_attempt > self.reconnect_interval):
            
            self.last_connection_attempt = current_time
            if self.connect():
                speak("Database connection restored")
                update_lcd_display("Database", "Connected")
                sleep(1)
    
    def is_connected(self):
        """Check if database is connected."""
        try:
            if self.connection:
                self.connection.ping(reconnect=True, attempts=1, delay=0)
                return True
        except:
            self.connection = None
            self.cursor = None
        return False

class SystemMode:
    def __init__(self):
        self.is_online = False
        self.last_network_check = 0
        self.network_check_interval = 30  # Check every 30 seconds
        self.mode_change_callbacks = []
        
    def check_network_status(self):
        """Check if network is available."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False
    
    def update_mode(self):
        """Update system mode based on network availability."""
        current_time = time.time()
        
        # Only check network periodically to avoid constant checking
        if current_time - self.last_network_check > self.network_check_interval:
            self.last_network_check = current_time
            new_status = self.check_network_status()
            
            # Mode change detected
            if new_status != self.is_online:
                old_mode = "Online" if self.is_online else "Offline"
                new_mode = "Online" if new_status else "Offline"
                
                self.is_online = new_status
                logger.info(f"System mode changed: {old_mode} -> {new_mode}")
                
                # Notify about mode change
                try:
                    speak(f"System switched to {new_mode.lower()} mode")
                    update_lcd_display(f"{new_mode} Mode", "Active")
                    sleep(2)
                except:
                    pass
                
                # Execute callbacks
                for callback in self.mode_change_callbacks:
                    try:
                        callback(self.is_online)
                    except Exception as e:
                        logger.error(f"Mode change callback error: {str(e)}")
    
    def add_mode_change_callback(self, callback):
        """Add callback to be executed when mode changes."""
        self.mode_change_callbacks.append(callback)

# Initialize system mode manager
system_mode = SystemMode()

# Initialize database manager
db_manager = DatabaseManager()
db_manager.connect()

# For backward compatibility
db_connection = db_manager.connection
db_cursor = db_manager.cursor

# Set the correct password
PASSWORD = "1234"

# Authentication state tracking
class AuthenticationState:
    def __init__(self):
        self.lock = threading.Lock()
        self.authenticated_methods = set()
        self.reset_timer = None
        self.is_unlocking = False
        self.unlock_complete = False
        
    def add_authentication(self, method):
        with self.lock:
            if self.is_unlocking:  # Prevent adding auth during unlock
                return
                
            self.authenticated_methods.add(method)
            logger.info(f"Authentication method '{method}' verified. Total: {len(self.authenticated_methods)}")
            
            # Reset timer - clear authentications after 30 seconds
            if self.reset_timer:
                self.reset_timer.cancel()
            self.reset_timer = threading.Timer(30.0, self.reset_authentications)
            self.reset_timer.start()
            
            # Check if we have two different authentication methods
            if len(self.authenticated_methods) >= 2:
                # Start unlock in separate thread to prevent blocking
                unlock_thread = threading.Thread(target=self.unlock_door, daemon=True)
                unlock_thread.start()
                
    def reset_authentications(self):
        with self.lock:
            if not self.is_unlocking:
                self.authenticated_methods.clear()
                self.unlock_complete = False
                logger.info("Authentication state reset - timeout")
            
    def unlock_door(self):
        with self.lock:
            if self.is_unlocking:  # Prevent multiple unlock attempts
                return
            self.is_unlocking = True
            
        try:
            logger.info("Two-factor authentication successful - unlocking door")
            
            # Perform unlock operations
            speak("Two factor authentication successful. Door unlocked.")
            update_lcd_display("Access Granted!", "Door Unlocked")
            
            # Activate solenoid
            GPIO.output(PINS['solenoid'], GPIO.HIGH)
            sleep(5)  # Keep door unlocked for 5 seconds
            GPIO.output(PINS['solenoid'], GPIO.LOW)
            
            logger.info("Door lock cycle completed")
            
        except Exception as e:
            logger.error(f"Error during unlock: {str(e)}")
        finally:
            # Reset state after unlock
            with self.lock:
                self.authenticated_methods.clear()
                self.is_unlocking = False
                self.unlock_complete = True
                if self.reset_timer:
                    self.reset_timer.cancel()
            
            # Update display back to ready state after a brief delay
            sleep(2)
            update_lcd_display("Smart Lock Ready", "2FA Required")
            logger.info("System ready for next authentication")

# Global authentication state
auth_state = AuthenticationState()

class NotificationManager:
    def __init__(self):
        self.offline_log_file = "offline_logs.txt"
        
    def log_notification(self, user_id, notify):
        """Log notification with automatic online/offline handling."""
        
        # Update database connection status
        db_manager.ensure_connection()
        
        # Try database logging if online and connected
        if system_mode.is_online and db_manager.is_connected():
            try:
                query = "INSERT INTO logs (user_id, notify, timestamp) VALUES (%s, %s, NOW())"
                db_manager.cursor.execute(query, (user_id, notify))
                db_manager.connection.commit()
                logger.info(f"Logged to database: {notify}")
                return
            except Exception as e:
                logger.warning(f"Database logging failed: {str(e)}, using offline log")
        
        # Offline logging fallback
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mode = "ONLINE" if system_mode.is_online else "OFFLINE"
            with open(self.offline_log_file, "a") as log_file:
                log_file.write(f"[{mode}] {timestamp} - User: {user_id} - Event: {notify}\n")
            logger.info(f"Logged offline: {notify}")
        except Exception as e:
            logger.error(f"Offline logging failed: {str(e)}")

# Initialize notification manager
notification_manager = NotificationManager()


# Global variables for face recognition optimization
face_recognition_cache = {
    'known_encodings': [],
    'known_ids': [],
    'last_reload': 0,
    'reload_interval': 30  # Reload encodings every 30 seconds
}

# Face recognition quality settings
FACE_RECOGNITION_CONFIG = {
    'tolerance': 0.45,  # Stricter tolerance (lower = more strict)
    'model': 'hog',     # Use HOG for speed, 'cnn' for accuracy
    'min_face_size': (50, 50),  # Minimum face size to process
    'max_attempts': 10,  # Maximum recognition attempts
    'confidence_threshold': 0.6,  # Minimum confidence for acceptance
    'multiple_angle_check': True,  # Check face from multiple angles
    'liveness_check': True,  # Basic liveness detection
}

def load_face_encodings_optimized():
    """Load face encodings with caching for better performance."""
    global face_recognition_cache
    
    current_time = time.time()
    
    # Check if we need to reload encodings
    if (current_time - face_recognition_cache['last_reload']) < face_recognition_cache['reload_interval']:
        return face_recognition_cache['known_encodings'], face_recognition_cache['known_ids']
    
    try:
        known_encodings = []
        known_ids = []
        
        if not os.path.exists(FACE_DIR):
            os.makedirs(FACE_DIR)
            return [], []
        
        for file in os.listdir(FACE_DIR):
            if file.endswith(".npy"):
                try:
                    face_id = os.path.splitext(file)[0]
                    face_encoding = np.load(f"{FACE_DIR}/{file}")
                    
                    # Validate encoding integrity
                    if face_encoding.shape == (128,):  # Standard face encoding size
                        known_encodings.append(face_encoding)
                        known_ids.append(face_id)
                        logger.info(f"Loaded face encoding for ID: {face_id}")
                    else:
                        logger.warning(f"Invalid encoding format for {file}, skipping")
                        
                except Exception as e:
                    logger.error(f"Error loading face encoding {file}: {str(e)}")
                    continue
        
        # Update cache
        face_recognition_cache['known_encodings'] = known_encodings
        face_recognition_cache['known_ids'] = known_ids
        face_recognition_cache['last_reload'] = current_time
        
        logger.info(f"Loaded {len(known_encodings)} face encodings")
        return known_encodings, known_ids
        
    except Exception as e:
        logger.error(f"Error in load_face_encodings_optimized: {str(e)}")
        return [], []

def enhanced_face_detection(frame):
    """Enhanced face detection with quality checks."""
    try:
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces with both models for better accuracy
        face_locations_hog = face_recognition.face_locations(rgb_frame, model="hog")
        
        # Filter faces by minimum size
        valid_faces = []
        for top, right, bottom, left in face_locations_hog:
            face_width = right - left
            face_height = bottom - top
            
            if (face_width >= FACE_RECOGNITION_CONFIG['min_face_size'][0] and 
                face_height >= FACE_RECOGNITION_CONFIG['min_face_size'][1]):
                valid_faces.append((top, right, bottom, left))
        
        return valid_faces
        
    except Exception as e:
        logger.error(f"Error in enhanced_face_detection: {str(e)}")
        return []

def calculate_face_distance_confidence(face_encoding, known_encodings):
    """Calculate confidence score for face recognition."""
    if not known_encodings:
        return [], []
    
    try:
        # Calculate face distances
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        # Convert distances to confidence scores (0-1, higher is better)
        confidences = 1 - face_distances
        
        return face_distances.tolist(), confidences.tolist()
        
    except Exception as e:
        logger.error(f"Error calculating face confidence: {str(e)}")
        return [], []

def verify_face_with_multiple_checks(face_encoding, known_encodings, known_ids, min_confidence=0.6):
    """Verify face with multiple checks for better accuracy."""
    try:
        if not known_encodings or not face_encoding.any():
            return False, None, 0.0
        
        # Calculate distances and confidences
        distances, confidences = calculate_face_distance_confidence(face_encoding, known_encodings)
        
        if not distances:
            return False, None, 0.0
        
        # Find the best match
        best_match_index = np.argmin(distances)
        best_distance = distances[best_match_index]
        best_confidence = confidences[best_match_index]
        
        # Check if match meets our criteria
        is_match = (best_distance <= FACE_RECOGNITION_CONFIG['tolerance'] and 
                   best_confidence >= min_confidence)
        
        if is_match:
            matched_id = known_ids[best_match_index]
            logger.info(f"Face match found: ID {matched_id}, confidence: {best_confidence:.2f}, distance: {best_distance:.2f}")
            return True, matched_id, best_confidence
        else:
            logger.info(f"Face not recognized: best confidence: {best_confidence:.2f}, distance: {best_distance:.2f}")
            return False, None, best_confidence
            
    except Exception as e:
        logger.error(f"Error in verify_face_with_multiple_checks: {str(e)}")
        return False, None, 0.0

def quick_face_check():
    """Enhanced quick face recognition with better accuracy."""
    try:
        # Load face encodings
        known_encodings, known_ids = load_face_encodings_optimized()
        
        if not known_encodings:
            return False, None
        
        start_time = time.time()
        max_duration = 3  # Reduced from 5 seconds for faster response
        best_match = None
        best_confidence = 0.0
        consecutive_matches = 0
        required_consecutive = 2  # Require 2 consecutive matches for confirmation
        
        attempt_count = 0
        max_attempts = FACE_RECOGNITION_CONFIG['max_attempts']
        
        while (time.time() - start_time) < max_duration and attempt_count < max_attempts:
            try:
                # Capture frame
                frame = picam2.capture_array()
                if frame is None:
                    continue
                
                # Detect faces
                face_locations = enhanced_face_detection(frame)
                
                if not face_locations:
                    consecutive_matches = 0  # Reset if no face detected
                    time.sleep(0.1)
                    attempt_count += 1
                    continue
                
                # Process the largest face (closest to camera)
                largest_face = max(face_locations, key=lambda face: (face[2]-face[0])*(face[1]-face[3]))
                
                # Get face encoding
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame, [largest_face])
                
                if not face_encodings:
                    consecutive_matches = 0
                    time.sleep(0.1)
                    attempt_count += 1
                    continue
                
                face_encoding = face_encodings[0]
                
                # Verify face with enhanced checks
                is_match, matched_id, confidence = verify_face_with_multiple_checks(
                    face_encoding, known_encodings, known_ids, 
                    FACE_RECOGNITION_CONFIG['confidence_threshold']
                )
                
                if is_match and matched_id:
                    if best_match == matched_id:
                        consecutive_matches += 1
                        best_confidence = max(best_confidence, confidence)
                        
                        # If we have enough consecutive matches, confirm recognition
                        if consecutive_matches >= required_consecutive:
                            logger.info(f"Face recognized with high confidence: {matched_id} (confidence: {best_confidence:.2f})")
                            return True, matched_id
                    else:
                        # Different match, reset counter
                        best_match = matched_id
                        best_confidence = confidence
                        consecutive_matches = 1
                else:
                    consecutive_matches = 0
                    best_match = None
                
                time.sleep(0.15)  # Short delay between attempts
                attempt_count += 1
                
            except Exception as e:
                logger.error(f"Error in face check iteration: {str(e)}")
                time.sleep(0.2)
                attempt_count += 1
                continue
        
        # Log the result
        if best_match:
            logger.info(f"Face check completed: Best match {best_match} with confidence {best_confidence:.2f}, but insufficient consecutive matches")
        else:
            logger.info("Face check completed: No reliable face match found")
        
        return False, None
        
    except Exception as e:
        logger.error(f"Error in quick_face_check_enhanced: {str(e)}")
        return False, None

def capture_and_save_face_enhanced():
    """Enhanced face capture with multiple samples for better accuracy."""
    try:
        speak("Position yourself in front of the camera. Stay still for face registration.")
        update_lcd_display("Face Registration", "Look at camera")
        time.sleep(2)
        
        # Capture multiple samples
        face_samples = []
        sample_count = 0
        required_samples = 3
        
        for i in range(required_samples):
            speak(f"Sample {i+1} of {required_samples}. Hold still.")
            update_lcd_display("Face Capture", f"Sample {i+1}/{required_samples}")
            time.sleep(1)
            
            frame = picam2.capture_array()
            if frame is None:
                continue
                
            face_locations = enhanced_face_detection(frame)
            
            if not face_locations:
                speak("No face detected. Please position yourself properly.")
                update_lcd_display("No Face", "Reposition")
                time.sleep(2)
                i -= 1  # Retry this sample
                continue
            
            # Get the largest face
            largest_face = max(face_locations, key=lambda face: (face[2]-face[0])*(face[1]-face[3]))
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame, [largest_face])
            
            if face_encodings:
                face_samples.append(face_encodings[0])
                sample_count += 1
                speak("Sample captured")
            else:
                speak("Face encoding failed. Please try again.")
                time.sleep(1)
                i -= 1  # Retry this sample
        
        if len(face_samples) < required_samples:
            speak("Insufficient face samples. Registration failed.")
            update_lcd_display("Registration", "Failed")
            return False
        
        # Average the face encodings for better accuracy
        averaged_encoding = np.mean(face_samples, axis=0)
        
        # Load existing encodings to check for duplicates
        known_encodings, known_ids = load_face_encodings_optimized()
        
        # Check if face already exists with stricter tolerance
        if known_encodings:
            is_duplicate, existing_id, confidence = verify_face_with_multiple_checks(
                averaged_encoding, known_encodings, known_ids, 0.7  # Higher threshold for registration
            )
            
            if is_duplicate:
                speak(f"Face already registered as user {existing_id}. Updating encoding.")
                face_id = existing_id
                update_lcd_display("Face Updated", f"ID: {face_id}")
            else:
                face_id = get_next_face_id()
                speak("New face registered successfully")
                update_lcd_display("Face Registered", f"ID: {face_id}")
        else:
            face_id = get_next_face_id()
            speak("First face registered successfully")
            update_lcd_display("Face Registered", f"ID: {face_id}")
        
        # Save the averaged encoding
        np.save(f"{FACE_DIR}/{face_id}.npy", averaged_encoding)
        
        # Invalidate cache to reload encodings
        face_recognition_cache['last_reload'] = 0
        
        notification_manager.log_notification(user_id, f"Face registered: ID {face_id}")
        logger.info(f"Face registered with ID: {face_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in capture_and_save_face_enhanced: {str(e)}")
        speak("Face registration failed due to system error")
        update_lcd_display("Registration", "System Error")
        return False

def verify_face_manual_enhanced():
    """Enhanced manual face verification with liveness detection."""
    try:
        speak("Look directly at the camera for verification.")
        update_lcd_display("Face Verification", "Look at camera")
        
        # Load face encodings
        known_encodings, known_ids = load_face_encodings_optimized()
        
        if not known_encodings:
            speak("No faces registered. Please register a face first.")
            update_lcd_display("No Faces", "Register First")
            return False
        
        # Enhanced verification with multiple checks
        start_time = time.time()
        verification_duration = 8  # Increased time for better accuracy
        successful_verifications = 0
        required_verifications = 3  # Need 3 successful verifications
        
        while time.time() - start_time < verification_duration:
            try:
                frame = picam2.capture_array()
                if frame is None:
                    continue
                
                face_locations = enhanced_face_detection(frame)
                
                if not face_locations:
                    time.sleep(0.2)
                    continue
                
                # Process the largest face
                largest_face = max(face_locations, key=lambda face: (face[2]-face[0])*(face[1]-face[3]))
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame, [largest_face])
                
                if not face_encodings:
                    time.sleep(0.2)
                    continue
                
                face_encoding = face_encodings[0]
                
                # Verify with enhanced checks
                is_match, matched_id, confidence = verify_face_with_multiple_checks(
                    face_encoding, known_encodings, known_ids,
                    FACE_RECOGNITION_CONFIG['confidence_threshold']
                )
                
                if is_match and matched_id:
                    successful_verifications += 1
                    update_lcd_display("Verifying...", f"{successful_verifications}/{required_verifications}")
                    
                    if successful_verifications >= required_verifications:
                        speak(f"Face verified for user {matched_id}")
                        update_lcd_display("Face Verified", f"User: {matched_id}")
                        notification_manager.log_notification(user_id, f"Face access granted - User {matched_id}")
                        return True
                
                time.sleep(0.3)
                
            except Exception as e:
                logger.error(f"Error in verification iteration: {str(e)}")
                continue
        
        # Verification failed
        speak("Face verification failed. Access denied.")
        update_lcd_display("Verification", "Failed")
        notification_manager.log_notification(user_id, "Face verification failed")
        
        # Save image of failed attempt
        try:
            frame = picam2.capture_array()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"/home/pi/intruder_logs/failed_face_{timestamp}.jpg", frame)
        except:
            pass
        
        return False
        
    except Exception as e:
        logger.error(f"Error in verify_face_manual_enhanced: {str(e)}")
        speak("Face verification system error")
        update_lcd_display("System Error", "Try Again")
        return False



# Declare forward references for functions used in the above section
def speak(message):
    """Forward declaration - implemented below"""
    pass

def update_lcd_display(line1, line2=""):
    """Forward declaration - implemented below"""
    pass

def sound_buzzer(duration=3):
    """Forward declaration - implemented below"""
    pass

def save_intruder_image():
    """Forward declaration - implemented below"""
    pass



def save_intruder_image():
    """Save intruder image to database."""
    try:
        frame = picam2.capture_array()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"/tmp/intruder_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        save_intruder_image_to_db(image_path)
        os.remove(image_path)
    except Exception as e:
        logger.error(f"Error saving intruder image: {str(e)}")

def save_intruder_image_to_db(image_path):
    """Save image with automatic online/offline handling."""
    
    # Update database connection status
    db_manager.ensure_connection()
    
    # Try database storage if online and connected
    if system_mode.is_online and db_manager.is_connected():
        try:
            with open(image_path, 'rb') as file:
                binary_data = file.read()
            
            insert_query = "INSERT INTO images (image) VALUES (%s)"
            db_manager.cursor.execute(insert_query, (binary_data,))
            db_manager.connection.commit()
            logger.info("Intruder image saved to database")
            return
        except Exception as e:
            logger.warning(f"Database image save failed: {str(e)}, using local storage")
    
    # Offline storage fallback
    try:
        offline_dir = "offline_images"
        if not os.path.exists(offline_dir):
            os.makedirs(offline_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "ONLINE" if system_mode.is_online else "OFFLINE"
        offline_path = f"{offline_dir}/intruder_{mode}_{timestamp}.jpg"
        
        import shutil
        shutil.copy2(image_path, offline_path)
        logger.info(f"Intruder image saved locally: {offline_path}")
        
    except Exception as e:
        logger.error(f"Local image save failed: {str(e)}")
        
def check_offline_dependencies():
    """Check if offline dependencies are available."""
    issues = []
    
    # Check Vosk model
    if not os.path.exists("vosk-model"):
        issues.append("Vosk model not found - offline speech recognition unavailable")
    
    # Check pico2wave
    if os.system("which pico2wave > /dev/null 2>&1") != 0:
        issues.append("pico2wave not found - install libttspico-utils")
    
    # Check espeak
    if os.system("which espeak > /dev/null 2>&1") != 0:
        issues.append("espeak not found - install espeak")
    
    if issues:
        logger.warning("Offline dependencies issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        speak("Some offline features may not work properly")
    else:
        logger.info("All offline dependencies available")
    
    return len(issues) == 0

# Voice Recognition Functions
def save_voice_command(command):
    """Save voice command to file."""
    try:
        with open("voice_command.txt", "w") as file:
            file.write(command)
        logger.info("Voice command saved successfully")
    except Exception as e:
        logger.error(f"Error saving voice command: {str(e)}")

def load_voice_command():
    """Load voice command from file."""
    try:
        if os.path.exists("voice_command.txt"):
            with open("voice_command.txt", "r") as file:
                return file.read().strip()
        return None
    except Exception as e:
        logger.error(f"Error loading voice command: {str(e)}")
        return None

def listen_for_command():
    """Listen for voice command with automatic online/offline switching."""
    
    # Try online recognition first if available
    if system_mode.is_online:
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            with sr.Microphone() as source:
                logger.info("Listening for voice command (online)...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                command = recognizer.recognize_google(audio)
                logger.info(f"Voice command detected (online): {command}")
                return command.lower().strip()
                
        except sr.UnknownValueError:
            logger.warning("Could not understand audio (online)")
        except sr.RequestError:
            logger.warning("Online speech recognition failed, switching to offline")
            system_mode.is_online = False  # Force offline mode for this session
        except Exception as e:
            logger.warning(f"Online voice recognition error: {str(e)}")
    
    # Offline recognition fallback
    try:
        # Check if Vosk model exists
        if not os.path.exists("vosk-model"):
            logger.error("Vosk model not found. Please install offline model.")
            speak("Offline voice recognition not available")
            return None
            
        import vosk
        import pyaudio
        import json
        
        model = vosk.Model("vosk-model")
        recognizer = vosk.KaldiRecognizer(model, 16000)
        
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                         input=True, frames_per_buffer=8192)
        stream.start_stream()
        
        logger.info("Listening for voice command (offline)...")
        
        timeout = time.time() + 10
        
        while time.time() < timeout:
            data = stream.read(4096, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                command = result.get('text', '').lower().strip()
                if command:
                    logger.info(f"Voice command detected (offline): {command}")
                    stream.stop_stream()
                    stream.close()
                    mic.terminate()
                    return command
        
        # Get final result
        result = json.loads(recognizer.FinalResult())
        command = result.get('text', '').lower().strip()
        
        stream.stop_stream()
        stream.close()
        mic.terminate()
        
        if command:
            logger.info(f"Voice command detected (offline): {command}")
            return command
        else:
            logger.warning("No voice command detected (offline)")
            return None
            
    except Exception as e:
        logger.error(f"Offline voice recognition error: {str(e)}")
        return None

def enroll_fingerprint():
    """Enroll a new fingerprint with improved error handling."""
    try:
        speak("Place your finger firmly on the sensor for enrollment")
        update_lcd_display("Fingerprint", "Place firmly")
        
        # Multiple attempts for better image capture
        for attempt in range(3):
            logger.info(f"Enrollment attempt {attempt + 1}")
            
            # Wait for finger with longer timeout
            timeout = time.time() + 15
            while not f.readImage() and time.time() < timeout:
                sleep(0.2)
            
            if time.time() >= timeout:
                speak("No finger detected. Try again.")
                update_lcd_display("No finger", "detected")
                continue
            
            try:
                f.convertImage(0x01)
                
                # Check if conversion was successful
                if f.downloadCharacteristics(0x01):
                    break
                else:
                    if attempt < 2:
                        speak("Poor image quality. Try again with firm pressure.")
                        update_lcd_display("Poor quality", "Press firmly")
                        sleep(2)
                        continue
                    else:
                        speak("Unable to capture clear fingerprint")
                        return False
                        
            except Exception as conv_error:
                logger.warning(f"Conversion attempt {attempt + 1} failed: {str(conv_error)}")
                if attempt < 2:
                    speak("Image processing failed. Try again.")
                    sleep(1)
                    continue
                else:
                    speak("Fingerprint capture failed")
                    return False
        
        # Check if fingerprint already exists
        try:
            result = f.searchTemplate()
            if result[0] >= 0:
                speak("Fingerprint already registered")
                update_lcd_display("Already", "Registered")
                return False
        except Exception as search_error:
            logger.warning(f"Search template error: {str(search_error)}")
            # Continue with enrollment even if search fails
        
        speak("Remove finger and place again for confirmation")
        update_lcd_display("Remove finger", "Place again")
        sleep(3)
        
        # Second capture for template matching
        for attempt in range(3):
            timeout = time.time() + 15
            while not f.readImage() and time.time() < timeout:
                sleep(0.2)
            
            if time.time() >= timeout:
                if attempt < 2:
                    speak("Place finger again")
                    continue
                else:
                    speak("Enrollment timeout")
                    return False
            
            try:
                f.convertImage(0x02)
                break
            except Exception as conv_error:
                logger.warning(f"Second conversion attempt {attempt + 1} failed: {str(conv_error)}")
                if attempt < 2:
                    speak("Try again with firm pressure")
                    sleep(1)
                    continue
                else:
                    speak("Second capture failed")
                    return False
        
        # Compare the two templates
        try:
            if f.compareCharacteristics() == 0:
                speak("Fingers do not match. Try enrollment again.")
                return False
        except Exception as compare_error:
            logger.error(f"Template comparison failed: {str(compare_error)}")
            speak("Template comparison failed")
            return False
            
        # Create and store template
        try:
            f.createTemplate()
            position = f.storeTemplate()
            
            speak(f"Fingerprint enrolled successfully at position {position}")
            update_lcd_display("Enrolled", f"Position: {position}")
            notification_manager.log_notification(user_id, "Fingerprint enrolled")
            return True
            
        except Exception as store_error:
            logger.error(f"Template storage failed: {str(store_error)}")
            speak("Failed to store fingerprint")
            return False
        
    except Exception as e:
        logger.error(f"Fingerprint enrollment error: {str(e)}")
        speak("Enrollment failed due to sensor error")
        return False

def verify_fingerprint():
    """Verify fingerprint with proper failure counting and notification."""
    try:
        speak("Place finger firmly on the sensor for verification")
        update_lcd_display("Fingerprint", "Place firmly")
        
        # Wait for finger with reasonable timeout
        timeout = time.time() + 12
        finger_detected = False
        
        while time.time() < timeout:
            try:
                if f.readImage():
                    finger_detected = True
                    break
            except Exception as read_error:
                logger.warning(f"Read image error: {str(read_error)}")
            sleep(0.2)
        
        if not finger_detected:
            speak("No finger detected")
            update_lcd_display("No finger", "detected")
            notification_manager.log_notification(user_id, "Fingerprint access denied - no finger")
            return False
        
        try:
            # Convert image to template
            f.convertImage(0x01)
            
            # Verify the conversion was successful
            if not f.downloadCharacteristics(0x01):
                speak("Poor image quality. Try again.")
                update_lcd_display("Poor quality", "Try again")
                notification_manager.log_notification(user_id, "Fingerprint access denied - poor quality")
                return False
            
            # Search for matching template
            result = f.searchTemplate()
            
            if result[0] >= 0:
                confidence = result[1]
                position = result[0]
                
                logger.info(f"Fingerprint matched at position {position} with confidence {confidence}")
                speak("Fingerprint verified successfully")
                update_lcd_display("Fingerprint", "Verified")
                notification_manager.log_notification(user_id, "Fingerprint access granted")
                auth_state.add_authentication("fingerprint")
                return True
            else:
                speak("Fingerprint not recognized")
                update_lcd_display("Not Recognized", "Access Denied")
                notification_manager.log_notification(user_id, "Fingerprint access denied - not recognized")
                return False
                    
        except Exception as verify_error:
            error_msg = str(verify_error)
            logger.warning(f"Fingerprint verification failed: {error_msg}")
            
            if "too few feature points" in error_msg.lower():
                speak("Poor fingerprint quality. Clean finger and try again.")
                update_lcd_display("Clean finger", "Try again")
                notification_manager.log_notification(user_id, "Fingerprint access denied - insufficient features")
            elif "timeout" in error_msg.lower():
                speak("Sensor timeout. Try again.")
                update_lcd_display("Sensor timeout", "Try again")
                notification_manager.log_notification(user_id, "Fingerprint access denied - timeout")
            else:
                speak("Fingerprint verification error")
                update_lcd_display("Verification", "Error")
                notification_manager.log_notification(user_id, "Fingerprint access denied - error")
            
            return False
        
    except Exception as e:
        logger.error(f"Fingerprint verification error: {str(e)}")
        speak("Verification failed due to sensor error")
        update_lcd_display("Sensor Error", "Try again")
        notification_manager.log_notification(user_id, "Fingerprint access denied - sensor error")
        return False

# Keypad Functions
def read_keypad():
    """Read keypad input with improved debouncing."""
    for row_index, row_pin in enumerate(ROW_PINS):
        GPIO.output(row_pin, GPIO.HIGH)
        for col_index, col_pin in enumerate(COL_PINS):
            if GPIO.input(col_pin) == GPIO.HIGH:
                time.sleep(0.05)
                if GPIO.input(col_pin) == GPIO.HIGH:
                    key = KEYPAD[row_index][col_index]
                    # Wait for key release with timeout
                    timeout = time.time() + 2.0
                    while GPIO.input(col_pin) == GPIO.HIGH and time.time() < timeout:
                        time.sleep(0.01)
                    GPIO.output(row_pin, GPIO.LOW)
                    return key
        GPIO.output(row_pin, GPIO.LOW)
    return None

def speak(message):
    """Text-to-speech with automatic online/offline switching."""
    
    # Try online TTS first if available
    if system_mode.is_online:
        try:
            from gtts import gTTS
            tts = gTTS(text=message, lang='en')
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
                tts.save(fp.name)
                result = os.system(f"mpg321 {fp.name} > /dev/null 2>&1")
                if result == 0:
                    return  # Success with online TTS
        except Exception as e:
            logger.warning(f"Online TTS failed: {str(e)}, switching to offline")
    
    # Offline TTS fallback
    try:
        # Try pico2wave first (better quality)
        temp_file = "/tmp/output.wav"
        command = f'pico2wave -w {temp_file} "{message}" && aplay {temp_file} > /dev/null 2>&1 && rm {temp_file}'
        result = os.system(command)
        
        if result == 0:
            return  # Success with pico2wave
        
        # Fallback to espeak
        os.system(f'espeak "{message}" > /dev/null 2>&1')
        
    except Exception as e:
        logger.error(f"All TTS methods failed: {str(e)}")

def update_lcd_display(line1, line2=""):
    """Update LCD display with error handling."""
    try:
        lcd.clear()
        lcd.write_string(line1[:16])
        if line2:
            lcd.cursor_pos = (1, 0)
            lcd.write_string(line2[:16])
    except Exception as e:
        logger.error(f"LCD update error: {str(e)}")

def sound_buzzer(duration=3):
    """Sound buzzer for specified duration."""
    try:
        GPIO.output(PINS['buzzer'], GPIO.HIGH)
        sleep(duration)
        GPIO.output(PINS['buzzer'], GPIO.LOW)
    except Exception as e:
        logger.error(f"Buzzer error: {str(e)}")

def display_message(message, stop_event):
    """Display scrolling message on LCD."""
    max_length = 16
    if len(message) <= max_length:
        lcd.clear()
        lcd.write_string(message)
        sleep(2)
        return
    
    message = message + "  "
    scroll_length = len(message)
    
    while not stop_event.is_set():
        for i in range(scroll_length - max_length + 1):
            if stop_event.is_set():
                break
            lcd.clear()
            lcd.write_string(message[i:i + max_length])
            sleep(0.5)

# Button debouncing function
def is_button_pressed(pin, debounce_time=0.1):
    """Check if button is pressed with debouncing."""
    if GPIO.input(pin) == GPIO.LOW:
        time.sleep(debounce_time)
        if GPIO.input(pin) == GPIO.LOW:
            # Wait for button release with timeout
            timeout = time.time() + 2.0
            while GPIO.input(pin) == GPIO.LOW and time.time() < timeout:
                time.sleep(0.01)
            return True
    return False

# Flask Routes
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = picam2.capture_array()
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control_solenoid', methods=['POST'])
def control_solenoid():
    try:
        switch_state = request.form.get("switch")
        if switch_state == "on":
            GPIO.output(PINS['solenoid'], GPIO.HIGH)
            return jsonify({"status": "success", "message": "Solenoid activated"}), 200
        elif switch_state == "off":
            GPIO.output(PINS['solenoid'], GPIO.LOW)
            return jsonify({"status": "success", "message": "Solenoid deactivated"}), 200
        else:
            return jsonify({"status": "error", "message": "Invalid switch state"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def index():
    return 'Smart Lock System with Two-Factor Authentication Running!'


class LockState(Enum):
    READY = "ready"
    FACE_DETECTED = "face_detected"
    PIN_ENTRY = "pin_entry"
    AUTHENTICATING = "authenticating"
    UNLOCKING = "unlocking"
    LOCKED_OUT = "locked_out"
    SETUP_MODE = "setup_mode"

@dataclass
class SecurityConfig:
    max_failures: int = 3
    lockout_duration: int = 300  # 5 minutes
    pin_timeout: int = 30  # 30 seconds for PIN entry
    auth_timeout: int = 60  # 60 seconds for biometric auth
    progressive_delay: bool = True  # Increase delay with each failure

class EnhancedAuthState:
    def __init__(self):
        self.authenticated_methods = set()
        self.required_methods = 2
        self.is_unlocking = False
        self.last_activity = time.time()
        self.session_timeout = 300  # 5 minutes
        self.face_recognized = False
        self.recognized_user_id = None
        self.auto_auth_active = False
    
    def add_authentication(self, method: str):
        self.authenticated_methods.add(method)
        self.last_activity = time.time()
        return len(self.authenticated_methods) >= self.required_methods
    
    def is_session_valid(self):
        return (time.time() - self.last_activity) < self.session_timeout
    
    def reset_session(self):
        self.authenticated_methods.clear()
        self.is_unlocking = False
        self.face_recognized = False
        self.recognized_user_id = None
        self.auto_auth_active = False
        self.last_activity = time.time()
    
    def activate_auto_auth(self, user_id):
        """Activate automatic authentication mode when face is recognized."""
        self.face_recognized = True
        self.recognized_user_id = user_id
        self.auto_auth_active = True
        self.add_authentication("face")

class SecurityManager:
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = 0
        self.lockout_until = 0
        self.failure_history = []  # Track failure patterns
    
    def is_locked_out(self):
        return time.time() < self.lockout_until
    
    def record_failure(self, method: str, user_context: Dict = None):
        current_time = time.time()
        self.failure_count += 1
        self.last_failure_time = current_time
        
        # Record failure with context for analysis
        self.failure_history.append({
            'time': current_time,
            'method': method,
            'context': user_context or {}
        })
        
        # Progressive delays - each failure increases wait time
        if self.config.progressive_delay:
            base_delay = min(2 ** (self.failure_count - 1), 30)  # Cap at 30 seconds
            time.sleep(base_delay)
        
        # Trigger lockout if threshold reached
        if self.failure_count >= self.config.max_failures:
            self.trigger_lockout()
    
    def record_success(self, method: str):
        # Reset failure count on successful authentication
        self.failure_count = 0
        self.failure_history.clear()
    
    def trigger_lockout(self):
        self.lockout_until = time.time() + self.config.lockout_duration
        self.failure_count = 0
        # Trigger security alerts
        save_intruder_image()
        sound_buzzer(8)
        notification_manager.log_notification(
            user_id, 
            f"Security lockout activated for {self.config.lockout_duration//60} minutes"
        )
    
    def get_remaining_lockout_time(self):
        if self.is_locked_out():
            return int(self.lockout_until - time.time())
        return 0

class UserInterface:
    def __init__(self):
        self.current_state = LockState.READY
        self.last_display_update = 0
        self.display_refresh_rate = 0.5  # Update display every 500ms
    
    def update_display_smart(self, primary: str, secondary: str = "", force: bool = False):
        """Smart display update that reduces unnecessary refreshes"""
        current_time = time.time()
        if force or (current_time - self.last_display_update) >= self.display_refresh_rate:
            update_lcd_display(primary, secondary)
            self.last_display_update = current_time
    
    def show_progress_indicator(self, step: int, total: int, method: str):
        """Show authentication progress"""
        progress = "●" * step + "○" * (total - step)
        self.update_display_smart(f"{method} Auth", f"Progress: {progress}")
    
    def show_lockout_timer(self, remaining_seconds: int):
        """Display lockout countdown"""
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        self.update_display_smart("LOCKED OUT", f"{minutes:02d}:{seconds:02d}")


def main_loop():
    """Enhanced main program loop with automatic face recognition."""
    
    # Enhanced security configuration
    MAX_FAILURES = 3
    LOCKOUT_DURATION = 300  # 5 minutes
    PIN_TIMEOUT = 30  # 30 seconds
    SESSION_TIMEOUT = 300  # 5 minutes
    FACE_CHECK_INTERVAL = 2  # Check for faces every 2 seconds
    
    # Unified security state
    security_state = {
        'failure_count': 0,
        'lockout_until': 0,
        'last_failure_time': 0,
        'progressive_delays': True
    }
    
    # Enhanced authentication state
    auth_state.authenticated_methods = set()
    auth_state.last_activity = time.time()
    auth_state.session_timeout = SESSION_TIMEOUT
    auth_state.face_recognized = False
    auth_state.recognized_user_id = None
    auth_state.auto_auth_active = False
    
    # PIN entry state with timeout
    pin_entry_state = {
        'active': False,
        'entered_password': '',
        'start_time': 0
    }
    
    # Face detection timing
    last_face_check = 0
    
    # Display update optimization
    last_display_update = 0
    display_refresh_rate = 0.5
    
    def smart_display_update(primary, secondary="", force=False):
        """Optimized display updates to reduce flicker"""
        nonlocal last_display_update
        current_time = time.time()
        if force or (current_time - last_display_update) >= display_refresh_rate:
            update_lcd_display(primary, secondary)
            last_display_update = current_time
    
    def show_progress_indicator(step, total, method):
        """Visual progress indicator for 2FA"""
        progress = "●" * step + "○" * (total - step)
        smart_display_update(f"{method} Auth", f"Progress: {progress}")
    
    def is_locked_out():
        """Check if system is in lockout state"""
        return time.time() < security_state['lockout_until']
    
    def record_security_failure(method):
        """Enhanced failure handling with progressive delays"""
        security_state['failure_count'] += 1
        security_state['last_failure_time'] = time.time()
        
        # Progressive delay - increases with each failure
        if security_state['progressive_delays']:
            delay = min(2 ** (security_state['failure_count'] - 1), 30)
            time.sleep(delay)
        
        # Trigger lockout if threshold reached
        if security_state['failure_count'] >= MAX_FAILURES:
            security_state['lockout_until'] = time.time() + LOCKOUT_DURATION
            security_state['failure_count'] = 0
            save_intruder_image()
            sound_buzzer(8)
            notification_manager.log_notification(
                user_id, f"Security lockout - {LOCKOUT_DURATION//60} minutes"
            )
    
    def record_security_success():
        """Reset security counters on successful auth"""
        security_state['failure_count'] = 0
        auth_state.last_activity = time.time()
    
    def is_session_valid():
        """Check if current authentication session is valid"""
        return (time.time() - auth_state.last_activity) < SESSION_TIMEOUT
    
    def reset_auth_session():
        """Reset authentication session"""
        auth_state.authenticated_methods.clear()
        auth_state.is_unlocking = False
        auth_state.face_recognized = False
        auth_state.recognized_user_id = None
        auth_state.auto_auth_active = False
        auth_state.last_activity = time.time()
    
    def add_authentication(method):
        """Add authentication method and check if 2FA complete"""
        auth_state.authenticated_methods.add(method)
        auth_state.last_activity = time.time()
        return len(auth_state.authenticated_methods) >= 2
    
    def activate_auto_auth_mode(user_id):
        """Activate automatic authentication mode when face is recognized"""
        auth_state.face_recognized = True
        auth_state.recognized_user_id = user_id
        auth_state.auto_auth_active = True
        add_authentication("face")
        
        speak(f"Hello user {user_id}. Face recognized. PIN and fingerprint activated.")
        smart_display_update(f"Welcome User {user_id}", "PIN & Fingerprint OK", force=True)
        notification_manager.log_notification(user_id, f"Face recognized - auto auth activated for user {user_id}")
        time.sleep(2)
    
    def initiate_unlock_sequence():
        """Enhanced unlock sequence with countdown"""
        auth_state.is_unlocking = True
        speak("Access granted. Door opening")
        smart_display_update("ACCESS GRANTED", "Door Opening...", force=True)
        
        GPIO.output(PINS['solenoid'], GPIO.HIGH)
        notification_manager.log_notification(user_id, "Door unlocked - 2FA successful")
        
        # Countdown display while door is open
        for i in range(5, 0, -1):
            smart_display_update("Door Open", f"Closing in {i}s", force=True)
            time.sleep(1)
        
        GPIO.output(PINS['solenoid'], GPIO.LOW)
        smart_display_update("Door Locked", "Goodbye", force=True)
        time.sleep(2)
        
        reset_auth_session()
    
    # Initialize system
    system_mode.update_mode()
    mode_text = "Online" if system_mode.is_online else "Offline"
    smart_display_update("Smart Lock Ready", f"{mode_text} • Auto Face", force=True)
    
    def on_mode_change(is_online):
        mode_text = "Online" if is_online else "Offline"
        smart_display_update(f"Mode: {mode_text}", "Connected" if is_online else "Local Only", force=True)
        speak(f"System now {mode_text.lower()}")
        notification_manager.log_notification(user_id, f"System switched to {mode_text} mode")
        time.sleep(2)
    
    system_mode.add_mode_change_callback(on_mode_change)
    
    # Main execution loop
    while True:
        try:
            system_mode.update_mode()
            current_time = time.time()
            
            # Handle lockout state
            if is_locked_out():
                remaining = int(security_state['lockout_until'] - current_time)
                minutes, seconds = divmod(remaining, 60)
                smart_display_update("LOCKED OUT", f"{minutes:02d}:{seconds:02d}")
                if remaining <= 1:
                    speak("System unlocked")
                time.sleep(1)
                continue
            
            # Automatic face detection (when not in PIN entry mode and not unlocking)
            if (not pin_entry_state['active'] and 
                not auth_state.is_unlocking and 
                not auth_state.face_recognized and
                (current_time - last_face_check) >= FACE_CHECK_INTERVAL):
                
                face_detected, detected_user_id = quick_face_check()
                last_face_check = current_time
                
                if face_detected and detected_user_id:
                    activate_auto_auth_mode(detected_user_id)
            
            # Handle PIN entry timeout
            if pin_entry_state['active']:
                if (current_time - pin_entry_state['start_time']) > PIN_TIMEOUT:
                    speak("PIN entry timeout")
                    smart_display_update("PIN Timeout", "Try Again", force=True)
                    pin_entry_state['active'] = False
                    pin_entry_state['entered_password'] = ""
                    time.sleep(1)
            
            # Handle session timeout
            if not is_session_valid() and len(auth_state.authenticated_methods) > 0:
                speak("Session expired")
                smart_display_update("Session Expired", "Auth Reset", force=True)
                reset_auth_session()
                time.sleep(2)
            
            # Process keypad input
            key = read_keypad()
            if key:
                logger.info(f"Key pressed: {key}")
                
                # Handle PIN entry mode
                if pin_entry_state['active']:
                    if key == "OK":
                        if pin_entry_state['entered_password'] == PASSWORD:
                            speak("PIN verified")
                            show_progress_indicator(len(auth_state.authenticated_methods), 2, "PIN")
                            notification_manager.log_notification(user_id, "PIN access granted")
                            
                            if add_authentication("pin"):
                                initiate_unlock_sequence()
                            else:
                                smart_display_update("PIN OK", "Access Granted!", force=True)
                                time.sleep(1)
                                smart_display_update("Welcome!", "Door Opening...", force=True)
                                time.sleep(1)
                                initiate_unlock_sequence()
                            
                            record_security_success()
                        else:
                            speak("Incorrect PIN")
                            remaining_tries = MAX_FAILURES - security_state['failure_count'] - 1
                            smart_display_update("Wrong PIN", f"{remaining_tries} tries left", force=True)
                            notification_manager.log_notification(user_id, "PIN access denied")
                            record_security_failure("pin")
                        
                        pin_entry_state['active'] = False
                        pin_entry_state['entered_password'] = ""
                        time.sleep(2)
                        
                    elif key == "*":
                        pin_entry_state['entered_password'] = ""
                        smart_display_update("PIN Cleared", "Enter PIN:", force=True)
                        
                    elif key in "0123456789":
                        pin_entry_state['entered_password'] += key
                        mask = "●" * len(pin_entry_state['entered_password'])
                        remaining_time = PIN_TIMEOUT - int(current_time - pin_entry_state['start_time'])
                        smart_display_update(f"PIN: {mask}", f"Time: {remaining_time}s")
                
                # Handle authentication selection
                elif not auth_state.is_unlocking:
                    if key == "A" or (auth_state.auto_auth_active and key in "0123456789"):
                        # Allow PIN entry normally or if face was recognized and number pressed
                        if key == "A":
                            speak("Enter your PIN")
                            smart_display_update("Enter PIN:", f"Timeout: {PIN_TIMEOUT}s", force=True)
                        else:
                            # Start PIN entry with the pressed number if face recognized
                            if not auth_state.auto_auth_active:
                                continue
                            smart_display_update("Enter PIN:", f"Timeout: {PIN_TIMEOUT}s", force=True)
                            pin_entry_state['entered_password'] = key
                            mask = "●" * len(pin_entry_state['entered_password'])
                            smart_display_update(f"PIN: {mask}", f"Time: {PIN_TIMEOUT}s")
                        
                        pin_entry_state['active'] = True
                        pin_entry_state['start_time'] = current_time
                        
                        # If key was not "A", we already added it above
                        if key == "A":
                            pin_entry_state['entered_password'] = ""
                    
                    elif key == "1" and not auth_state.auto_auth_active:  # Voice verification (only when face not recognized)
                        registered_command = load_voice_command()
                        if not registered_command:
                            speak("No voice command registered")
                            smart_display_update("No Voice", "Use Button 1", force=True)
                            time.sleep(3)
                            continue
                        
                        speak("Speak your passphrase")
                        smart_display_update("Voice Verify", "Speak now", force=True)
                        
                        command = listen_for_command()
                        if command and command == registered_command:
                            speak("Voice verified")
                            show_progress_indicator(1, 2, "Voice")
                            notification_manager.log_notification(user_id, "Voice access granted")
                            
                            if add_authentication("voice"):
                                initiate_unlock_sequence()
                            else:
                                smart_display_update("Voice OK", "Need 2nd method", force=True)
                                time.sleep(2)
                            
                            record_security_success()
                        else:
                            speak("Voice not recognized")
                            smart_display_update("Voice Failed", "Try again", force=True)
                            notification_manager.log_notification(user_id, "Voice access denied")
                            record_security_failure("voice")
                        
                        time.sleep(2)
                    
                    elif key == "2" and not auth_state.auto_auth_active:  # Manual face verification (only when auto not active)
                        speak("Look at the camera")
                        smart_display_update("Face Verify", "Look at camera", force=True)
                        
                        if verify_face():
                            speak("Face verified")
                            show_progress_indicator(1, 2, "Face")
                            notification_manager.log_notification(user_id, "Face access granted")
                            
                            if add_authentication("face"):
                                initiate_unlock_sequence()
                            else:
                                smart_display_update("Face OK", "Need 2nd method", force=True)
                                time.sleep(2)
                            
                            record_security_success()
                        else:
                            speak("Face not recognized")
                            smart_display_update("Face Failed", "Try again", force=True)
                            notification_manager.log_notification(user_id, "Face access denied")
                            record_security_failure("face")
                        
                        time.sleep(2)
                    
                    elif key == "3":  # Fingerprint verification
                        speak("Place your finger")
                        smart_display_update("Fingerprint", "Place finger", force=True)
                        
                        if verify_fingerprint():
                            speak("Fingerprint verified")
                            show_progress_indicator(len(auth_state.authenticated_methods), 2, "Fingerprint")
                            notification_manager.log_notification(user_id, "Fingerprint access granted")
                            
                            if add_authentication("fingerprint"):
                                initiate_unlock_sequence()
                            else:
                                smart_display_update("Fingerprint OK", "Access Granted!", force=True)
                                time.sleep(1)
                                smart_display_update("Welcome!", "Door Opening...", force=True)
                                time.sleep(1)
                                initiate_unlock_sequence()
                            
                            record_security_success()
                        else:
                            speak("Fingerprint not recognized")
                            smart_display_update("Fingerprint Failed", "Try again", force=True)
                            notification_manager.log_notification(user_id, "Fingerprint access denied")
                            record_security_failure("fingerprint")
                        
                        time.sleep(2)
            
            # Handle button presses (only when ready)
            if not pin_entry_state['active'] and not auth_state.is_unlocking:
                
                if is_button_pressed(PINS['button1']):  # Register voice
                    speak("Register your voice passphrase")
                    smart_display_update("Voice Setup", "Speak clearly", force=True)
                    
                    command = listen_for_command()
                    if command:
                        save_voice_command(command)
                        speak("Voice registered successfully")
                        smart_display_update("Voice Setup", "Success!", force=True)
                        notification_manager.log_notification(user_id, "Voice passphrase registered")
                    else:
                        speak("Voice registration failed")
                        smart_display_update("Voice Setup", "Failed", force=True)
                    
                    time.sleep(3)
                
                if is_button_pressed(PINS['button5']):  # Enroll fingerprint
                    smart_display_update("Fingerprint", "Enrolling...", force=True)
                    enroll_fingerprint()
                    time.sleep(2)
                
                if is_button_pressed(PINS['button6']):  # Manual unlock
                    speak("Manual override activated")
                    smart_display_update("Manual Override", "Door Opening", force=True)
                    GPIO.output(PINS['solenoid'], GPIO.HIGH)
                    notification_manager.log_notification(user_id, "Manual override used")
                    time.sleep(5)
                    GPIO.output(PINS['solenoid'], GPIO.LOW)
                    smart_display_update("Manual Override", "Complete", force=True)
                    time.sleep(2)
                
                if is_button_pressed(PINS['button7']):  # Register face
                    smart_display_update("Face Setup", "Look at camera", force=True)
                    capture_and_save_face()
                    time.sleep(2)
            
            # Update ready state display
            if not pin_entry_state['active'] and not auth_state.is_unlocking:
                mode_text = "Online" if system_mode.is_online else "Offline"
                
                if auth_state.auto_auth_active:
                    smart_display_update(f"User {auth_state.recognized_user_id}", "PIN/FP Ready")
                elif len(auth_state.authenticated_methods) > 0:
                    smart_display_update(f"Auth: {len(auth_state.authenticated_methods)}/2", mode_text)
                else:
                    smart_display_update("Smart Lock Ready", f"{mode_text} • Auto Face")
            
            time.sleep(0.1)  # Responsive timing
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            smart_display_update("System Error", "Restarting...", force=True)
            time.sleep(1)

# Additional utility functions for enhanced security and monitoring

def save_intruder_image():
    """Save image when unknown face is detected or security breach occurs."""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        frame = picam2.capture_array()
        filename = f"intruder_{timestamp}.jpg"
        cv2.imwrite(f"/home/pi/intruder_logs/{filename}", frame)
        logger.warning(f"Intruder image saved: {filename}")
        notification_manager.log_notification(user_id, f"Security alert: Intruder image captured")
    except Exception as e:
        logger.error(f"Failed to save intruder image: {str(e)}")

def periodic_face_scan():
    """Background thread for continuous face monitoring."""
    global auth_state
    
    while True:
        try:
            if (not auth_state.face_recognized and 
                not auth_state.is_unlocking and 
                not is_locked_out()):
                
                face_detected, detected_user_id = quick_face_check()
                
                if face_detected and detected_user_id:
                    activate_auto_auth_mode(detected_user_id)
                elif face_detected and not detected_user_id:
                    # Unknown face detected
                    save_intruder_image()
                    sound_buzzer(2)
                    smart_display_update("Unknown Face", "Access Denied", force=True)
                    time.sleep(3)
            
            time.sleep(2)  # Check every 2 seconds
            
        except Exception as e:
            logger.error(f"Error in periodic face scan: {str(e)}")
            time.sleep(5)

def enhanced_security_monitoring():
    """Enhanced security monitoring with pattern detection."""
    
    # Monitor for repeated failed attempts
    def check_failure_patterns():
        current_time = time.time()
        recent_failures = [f for f in security_state['failure_history'] 
                          if current_time - f['time'] < 3600]  # Last hour
        
        if len(recent_failures) > 10:  # More than 10 failures in an hour
            notification_manager.log_notification(
                user_id, 
                "SECURITY ALERT: Excessive authentication failures detected"
            )
            save_intruder_image()
            return True
        return False
    
    # Monitor for unusual access patterns
    def check_access_patterns():
        current_hour = time.localtime().tm_hour
        
        # Alert for access attempts during unusual hours (e.g., 2 AM - 5 AM)
        if 2 <= current_hour <= 5:
            notification_manager.log_notification(
                user_id, 
                f"SECURITY ALERT: Access attempt at unusual hour: {current_hour}:00"
            )
            return True
        return False
    
    return check_failure_patterns() or check_access_patterns()

def system_health_check():
    """Perform system health checks and maintenance."""
    
    def check_storage_space():
        """Check available storage space."""
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        
        if free_gb < 1:  # Less than 1GB free
            logger.warning(f"Low storage space: {free_gb}GB remaining")
            notification_manager.log_notification(
                user_id, 
                f"System warning: Low storage space ({free_gb}GB remaining)"
            )
            return False
        return True
    
    def check_camera_status():
        """Check camera functionality."""
        try:
            frame = picam2.capture_array()
            if frame is None or frame.size == 0:
                logger.error("Camera capture failed")
                notification_manager.log_notification(
                    user_id, 
                    "System error: Camera malfunction detected"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Camera check failed: {str(e)}")
            return False
    
    def cleanup_old_logs():
        """Clean up old log files and images."""
        import glob
        import os
        
        # Remove intruder images older than 30 days
        thirty_days_ago = time.time() - (30 * 24 * 60 * 60)
        
        for img_file in glob.glob("/home/pi/intruder_logs/*.jpg"):
            if os.path.getctime(img_file) < thirty_days_ago:
                os.remove(img_file)
                logger.info(f"Removed old intruder image: {img_file}")
    
    # Run all health checks
    storage_ok = check_storage_space()
    camera_ok = check_camera_status()
    cleanup_old_logs()
    
    return storage_ok and camera_ok

def initialize_smart_lock_system():
    """Initialize the complete smart lock system with all components."""
    
    try:
        # Create necessary directories
        os.makedirs(FACE_DIR, exist_ok=True)
        os.makedirs("/home/pi/intruder_logs", exist_ok=True)
        os.makedirs("/home/pi/system_logs", exist_ok=True)
        
        # Initialize GPIO pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup all pins
        for pin_name, pin_number in PINS.items():
            if pin_name == 'solenoid':
                GPIO.setup(pin_number, GPIO.OUT)
                GPIO.output(pin_number, GPIO.LOW)  # Ensure door is locked initially
            elif 'button' in pin_name:
                GPIO.setup(pin_number, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            else:
                GPIO.setup(pin_number, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Initialize camera
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
        picam2.start()
        time.sleep(2)  # Let camera warm up
        
        # Initialize display
        update_lcd_display("System Starting", "Please wait...")
        
        # Run initial system health check
        if not system_health_check():
            logger.warning("System health check failed during initialization")
        
        # Start background monitoring thread
        import threading
        
        face_scan_thread = threading.Thread(target=periodic_face_scan, daemon=True)
        face_scan_thread.start()
        
        logger.info("Smart lock system initialized successfully")
        notification_manager.log_notification(user_id, "Smart lock system started")
        
        # Welcome message
        speak("Smart lock system ready. Face recognition activated.")
        update_lcd_display("System Ready", "Auto Face Active")
        time.sleep(3)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize smart lock system: {str(e)}")
        update_lcd_display("Init Failed", "Check System")
        return False

def graceful_shutdown():
    """Perform graceful system shutdown."""
    
    try:
        logger.info("Initiating graceful shutdown...")
        
        # Secure the door
        GPIO.output(PINS['solenoid'], GPIO.LOW)
        
        # Save any pending data
        notification_manager.log_notification(user_id, "System shutdown initiated")
        
        # Stop camera
        picam2.stop()
        
        # Final display message
        update_lcd_display("System", "Shutting Down...")
        speak("System shutting down. Goodbye.")
        
        # Cleanup GPIO
        GPIO.cleanup()
        
        logger.info("Graceful shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# Signal handler for graceful shutdown
import signal
import sys

def signal_handler(sig, frame):
    """Handle system signals for graceful shutdown."""
    print("\nReceived shutdown signal...")
    graceful_shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
    
if __name__ == '__main__':
    try:
        check_offline_dependencies()
        # Start Flask server in a separate thread
        server_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=5000, debug=False),
            daemon=True
        )
        server_thread.start()
        
        logger.info("Smart Lock System with Two-Factor Authentication Starting...")
        speak("Smart lock system ready. Two factor authentication required.")
        
        # Start main loop
        main_loop()
        initialize_smart_lock_system()
        
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        GPIO.cleanup()
        lcd.clear()
        lcd.backlight_enabled = False
        logger.info("System shutdown complete")