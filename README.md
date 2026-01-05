# Smart Door Lock System V2

A comprehensive multi-platform smart door lock system featuring biometric authentication, remote access control, and real-time monitoring capabilities.

##  Architecture Overview

This system consists of four main components:

### 1. Database Layer (`database/`)
- **MySQL Database**: `smartdb.sql`
- **Tables**: users, logs, images, default settings
- Stores user credentials, access logs, captured images, and system configuration

### 2. Raspberry Pi Controller (`raspberryPythonSC/`)
- **Core Controller**: `rei3.py`
- **Hardware Integration**: GPIO control, fingerprint sensor, LCD display, camera
- **Biometric Features**: Fingerprint recognition, face recognition, voice commands
- **API Server**: Flask-based REST API for remote communication
- **Security Features**: Multi-factor authentication, real-time monitoring

### 3. Mobile/Desktop Application (`smartlockV2SCMobileDeskWebApp/`)
- **Framework**: .NET MAUI (Multi-platform App UI)
- **Platforms**: Android, iOS, Windows, macOS
- **Features**: User management, remote lock control, notifications, biometric registration
- **UI**: Modern XAML-based interface with theming support

### 4. Web Application (`smartlockV2SCWebApp/`)
- **Backend**: PHP with MySQL
- **Frontend**: HTML/CSS/JavaScript
- **Features**: Admin dashboard, user approval, notification management, image gallery
- **Security**: Session-based authentication, role-based access control

##  Features

###  Authentication Methods
- **Biometric**: Fingerprint and facial recognition
- **Voice Commands**: Speech-to-text access control
- **PIN/Keypad**: Physical keypad input
- **Remote Access**: Mobile app and web interface

###  User Interfaces
- **Mobile App**: Native experience across platforms
- **Web Dashboard**: Browser-based admin interface
- **LCD Display**: Local status and feedback
- **Voice Feedback**: Audio confirmations and alerts

###  Monitoring & Notifications
- **Real-time Logs**: Access attempt tracking
- **Email Alerts**: Security notifications
- **Push Notifications**: Mobile alerts for events
- **Image Capture**: Security camera integration

###  Administration
- **User Management**: Add, approve, and manage users
- **Access Control**: Role-based permissions
- **System Configuration**: Remote settings adjustment
- **Audit Trail**: Comprehensive logging

##  System Requirements

### Hardware Requirements
- **Raspberry Pi** (4B recommended) with camera module
- **Fingerprint Sensor** (compatible with PyFingerprint)
- **LCD Display** (I2C interface)
- **GPIO-connected components**: Keypad, relay for lock control
- **Microphone** and **Speaker** for voice features

### Software Requirements
- **Python 3.8+** with required libraries
- **MySQL 8.0+** database server
- **.NET 7.0+** SDK for MAUI app
- **PHP 8.0+** with MySQL extension
- **Apache/Nginx** web server

##  Installation & Setup

### 1. Database Setup
```bash
# Create MySQL database
mysql -u root -p < database/smartdb.sql
```

### 2. Raspberry Pi Controller Setup
```bash
cd raspberryPythonSC

# Install Python dependencies
pip install -r requirements.txt

# Configure GPIO pins and sensors
# Edit rei3.py for your specific hardware configuration

# Run the controller
python rei3.py
```

### 3. Mobile/Desktop App Setup
```bash
cd smartlockV2SCMobileDeskWebApp

# Restore NuGet packages
dotnet restore

# Build for specific platform
dotnet build -f net7.0-android    # For Android
dotnet build -f net7.0-ios        # For iOS
dotnet build -f net7.0-windows    # For Windows
```

### 4. Web Application Setup
```bash
cd smartlockV2SCWebApp

# Configure database connection in dbconnect.php
# Upload files to web server
# Ensure proper permissions for image/ directory
```

##  Configuration

### Database Configuration
Update connection settings in:
- `smartlockV2SCWebApp/dbconnect.php`
- `raspberryPythonSC/rei3.py` (MySQL connection parameters)

### Hardware Configuration
Modify GPIO pin assignments in `raspberryPythonSC/rei3.py`:
```python
ROW_PINS = [5, 27, 17, 4]
COL_PINS = [7, 8, 25, 18]
```

### API Endpoints
The Flask server runs on port 5000 by default. Update URLs in mobile/web apps accordingly.

##  Usage

### Starting the System
1. **Start Database**: Ensure MySQL is running
2. **Launch Controller**: Run `python rei3.py` on Raspberry Pi
3. **Deploy Web App**: Host PHP files on web server
4. **Install Mobile App**: Build and install MAUI app

### User Registration
1. **Admin Approval**: Register via web interface or mobile app
2. **Biometric Enrollment**: Capture fingerprint/face data
3. **Access Grant**: Admin approves user access

### Access Control
- **Local Access**: Use keypad, fingerprint, or voice commands
- **Remote Access**: Control via mobile app or web interface
- **Emergency Access**: Default PIN for backup access

##  Security Features

- **Multi-factor Authentication**: Combine multiple verification methods
- **Encrypted Communications**: Secure API endpoints
- **Access Logging**: Track all authentication attempts
- **Intrusion Detection**: Alert on suspicious activities
- **Session Management**: Secure web and mobile sessions

##  API Reference

### Flask API Endpoints (Raspberry Pi)
- `GET /status` - System status
- `POST /unlock` - Remote unlock
- `POST /register_biometric` - Register biometric data
- `GET /logs` - Access logs
- `POST /capture_image` - Capture security image

### Web API Endpoints
- `/userreg.php` - User registration
- `/userlog.php` - User login
- `/toggle_lock.php` - Lock control
- `/allnotif.php` - Notifications
- `/images.php` - Image management

##  Troubleshooting

### Common Issues
- **GPIO Permission Denied**: Run with `sudo` or configure permissions
- **Camera Not Working**: Check camera module connection and enable in raspi-config
- **Database Connection Failed**: Verify MySQL credentials and network access
- **Fingerprint Sensor Error**: Check sensor wiring and power supply

### Logs and Debugging
- **Python Logs**: Check console output from `rei3.py`
- **Web Logs**: Check Apache/Nginx error logs
- **Database Logs**: Monitor MySQL error logs

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Authors

- **Project Team** - Initial development and implementation

##  Acknowledgments

- Raspberry Pi Foundation for hardware platform
- .NET MAUI team for cross-platform framework
- Open source libraries and communities

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with local laws and regulations regarding access control systems.