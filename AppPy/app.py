import os
import time
import json
import logging
from flask import Flask, render_template, request, redirect, url_for, abort, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import face_recognition
from firebase_admin import credentials, firestore, initialize_app
import paho.mqtt.client as mqtt
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import requests  # Importar aquí para evitar problemas si no se usa en otros endpoints

# -----------------------------
# Cargar Variables de Entorno
# -----------------------------
load_dotenv()  # Cargar variables desde un archivo .env

# -----------------------------
# Configuración Inicial de Flask
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback_secret_key')  # Cargar desde .env o usar valor por defecto

# Configurar Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Nombre de la función de login

# Configurar Flask-SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Configurar el logging para la aplicación
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("server.log"),  # Archivo de log
                        logging.StreamHandler()             # Salida a la consola
                    ])
logger = logging.getLogger()

# -----------------------------
# Configuración de Firebase
# -----------------------------
# Ruta al archivo de credenciales de Firebase
FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH', 'credentials/serviceAccountKey.json')

# Inicializar Firebase
cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
initialize_app(cred)
db = firestore.client()

# -----------------------------
# Definición de Modelos
# -----------------------------
# Ya no necesitas SQLAlchemy para SensorData si usas Firebase
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id  # Document ID en Firestore
        self.username = username
        self.password_hash = password_hash

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# -----------------------------
# Configuración MQTT
# -----------------------------
MQTT_BROKER = os.getenv('MQTT_BROKER', '35.238.55.111')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_USER = os.getenv('MQTT_USER', 'aldo6868')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD', 'Tapia5307=')
MQTT_CLIENT_ID = os.getenv('MQTT_CLIENT_ID', 'flask_server')
MQTT_TOPIC_COMMANDS = os.getenv('MQTT_TOPIC_COMMANDS', 'car/commands')
MQTT_TOPIC_SET_POWER = os.getenv('MQTT_TOPIC_SET_POWER', 'car/set_power')
MQTT_TOPIC_SENSOR_DATA = os.getenv('MQTT_TOPIC_SENSOR_DATA', 'car/sensor_data')

# Crear cliente MQTT
mqtt_client = mqtt.Client(MQTT_CLIENT_ID)

# Configurar credenciales MQTT
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

# Definir callbacks MQTT
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Conectado al broker MQTT.")
        # Suscribirse a los tópicos necesarios
        client.subscribe(MQTT_TOPIC_COMMANDS)
        client.subscribe(MQTT_TOPIC_SET_POWER)
        client.subscribe(MQTT_TOPIC_SENSOR_DATA)
    else:
        logger.error(f"Error al conectar al broker MQTT, código de retorno {rc}")

def on_message(client, userdata, msg):
    try:
        topic_decoded = msg.topic  # msg.topic ya es una cadena (str)

        # Verificar si msg.payload es de tipo bytes antes de decodificar
        if isinstance(msg.payload, bytes):
            payload_decoded = msg.payload.decode()
        else:
            payload_decoded = msg.payload  # Ya es una cadena

        logger.info(f"Mensaje recibido vía MQTT en '{topic_decoded}': {payload_decoded}")

        if topic_decoded == MQTT_TOPIC_COMMANDS or topic_decoded == MQTT_TOPIC_SET_POWER:
            # Emitir comando a través de SocketIO
            socketio.emit('mqtt_command', {'topic': topic_decoded, 'command': payload_decoded})
        elif topic_decoded == MQTT_TOPIC_SENSOR_DATA:
            # Procesar y almacenar datos de sensores
            process_sensor_data(payload_decoded)
    except Exception as e:
        logger.error(f"Error al procesar el mensaje MQTT: {e}")

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Manejo de reconexión MQTT con reintentos
MAX_RETRIES = 5
RETRY_DELAY = 5  # Segundos

for attempt in range(1, MAX_RETRIES + 1):
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        logger.info("Intento de conexión MQTT exitoso.")
        break
    except Exception as e:
        logger.error(f"Intento {attempt} de conexión MQTT fallido: {e}")
        if attempt == MAX_RETRIES:
            logger.critical("Máximos intentos de conexión MQTT alcanzados. Saliendo de la aplicación.")
            exit(1)
        else:
            logger.info(f"Reintentando en {RETRY_DELAY} segundos...")
            time.sleep(RETRY_DELAY)

# Iniciar el loop MQTT en un hilo separado
mqtt_client.loop_start()

# -----------------------------
# Carga de Rostros Conocidos
# -----------------------------
KNOWN_FACES_DIR = 'known_faces'
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    if not os.path.exists(KNOWN_FACES_DIR):
        logger.warning(f"La carpeta {KNOWN_FACES_DIR} no existe.")
        return
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                # Extraer el nombre de la imagen (por ejemplo, juan_perez.jpg -> Juan Perez)
                name = os.path.splitext(filename)[0].replace('_', ' ').title()
                known_face_names.append(name)
                logger.info(f"Cargado rostro conocido: {name}")
            else:
                logger.warning(f"No se pudo encontrar un rostro en la imagen {filename}")

load_known_faces()

# -----------------------------
# Rutas de la Aplicación
# -----------------------------

@login_manager.user_loader
def load_user(user_id):
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            return User(id=user_doc.id, username=user_data['username'], password_hash=user_data['password_hash'])
        else:
            return None
    except Exception as e:
        logger.error(f"Error al cargar el usuario: {e}")
        return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        logger.info(f"Intentando iniciar sesión para el usuario: {username}")

        if not username or not password:
            logger.warning("Intento de inicio de sesión con campos vacíos.")
            return render_template('login.html', error="Todos los campos son obligatorios.")

        try:
            users_ref = db.collection('users')
            query = users_ref.where('username', '==', username).stream()
            user = None
            for doc in query:
                user_data = doc.to_dict()
                user = User(id=doc.id, username=user_data['username'], password_hash=user_data['password_hash'])
                break

            if user:
                logger.info(f"Usuario encontrado: {username}. Verificando contraseña.")
                if user.check_password(password):
                    login_user(user)
                    logger.info(f"Usuario {username} inició sesión exitosamente.")
                    return redirect(url_for('menu'))
                else:
                    logger.warning(f"Contraseña incorrecta para el usuario: {username}")
                    return render_template('login.html', error="Usuario o contraseña incorrectos.")
            else:
                logger.warning(f"Usuario no encontrado: {username}")
                return render_template('login.html', error="Usuario o contraseña incorrectos.")
        except Exception as e:
            logger.error(f"Error al autenticar usuario: {e}")
            return render_template('login.html', error="Error interno. Inténtalo de nuevo.")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    logger.info("Usuario cerró sesión.")
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return redirect(url_for('menu'))

@app.route('/menu.html')
@login_required
def menu():
    return render_template('menu.html')

@app.route('/gps.html')
@login_required
def gps():
    return render_template('gps.html')

@app.route('/objetos_reconocidos.html')
@login_required
def objetos_reconocidos():
    return render_template('objetos_reconocidos.html')

@app.route('/historial.html')
@login_required
def historial_view():
    return render_template('historial.html')

# -----------------------------
# Clase Historial
# -----------------------------
class Historial:
    def __init__(self, db_client):
        self.db_client = db_client

    def obtener_datos_sensor(self, limite=100):
        """
        Recupera los últimos 'limite' registros de sensor_data y mapea los nombres de los campos.
        """
        try:
            sensores_ref = self.db_client.collection('sensor_data').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limite)
            sensores_docs = sensores_ref.stream()
            sensores_mapeados = []
            for doc in sensores_docs:
                sensor = doc.to_dict()
                sensores_mapeados.append({
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sensor.get('timestamp', time.time()))),
                    "ultrasonico_distancia": sensor.get('sensor_ultrasonico', "N/A"),
                    "ultrasonico_digital_1": "Detectado" if sensor.get('ultrasonido_1') == 0 else "Libre",
                    "ultrasonico_digital_2": "Detectado" if sensor.get('ultrasonido_2') == 0 else "Libre",
                    "ultrasonico_digital_3": "Detectado" if sensor.get('ultrasonido_3') == 0 else "Libre",
                    "sensor_ir": "Detectado" if sensor.get('sensor_ir') == 0 else "No Detectado",
                    "sensor_fuego_1": "Detectado" if sensor.get('sensor_fuego_1') == 0 else "No Detectado",
                    "sensor_fuego_2": "Detectado" if sensor.get('sensor_fuego_2') == 0 else "No Detectado",
                    "gps_lat": sensor.get('gps_lat', "N/A"),
                    "gps_lng": sensor.get('gps_lng', "N/A")
                })
            return sensores_mapeados
        except Exception as e:
            logger.error(f"Error al obtener datos de sensor: {e}")
            return []

# -----------------------------
# API Endpoints para Control del Carrito
# -----------------------------

@app.route('/send_command', methods=['POST'])
@login_required
def send_command():
    data = request.get_json()
    command = data.get('command')
    if command:
        if mqtt_client.is_connected():
            mqtt_client.publish(MQTT_TOPIC_COMMANDS, command)
            logger.info(f"[Flask] Comando enviado a MQTT: {command}")
            return jsonify({'status': 'success', 'message': 'Comando enviado'}), 200
        else:
            logger.error("[Flask] No conectado al broker MQTT")
            return jsonify({'status': 'error', 'message': 'No conectado al broker MQTT'}), 500
    else:
        logger.warning("[Flask] No se recibió ningún comando en /send_command")
        return jsonify({'status': 'error', 'message': 'No se recibió ningún comando'}), 400

@app.route('/set_power', methods=['POST'])
@login_required
def set_power():
    data = request.get_json()
    power = data.get('power')
    if power:
        try:
            power_value = int(power)
            if 0 <= power_value <= 65535:
                if mqtt_client.is_connected():
                    mqtt_client.publish(MQTT_TOPIC_SET_POWER, str(power_value))
                    logger.info(f"[Flask] Potencia enviada a MQTT: {power_value}")
                    return jsonify({'status': 'success', 'message': 'Potencia enviada'}), 200
                else:
                    logger.error("[Flask] No conectado al broker MQTT")
                    return jsonify({'status': 'error', 'message': 'No conectado al broker MQTT'}), 500
            else:
                logger.warning("[Flask] Valor de potencia fuera de rango en /set_power")
                return jsonify({'status': 'error', 'message': 'Valor de potencia fuera de rango (0-65535)'}), 400
        except ValueError:
            logger.warning("[Flask] Valor de potencia inválido en /set_power")
            return jsonify({'status': 'error', 'message': 'Valor de potencia inválido'}), 400
    else:
        logger.warning("[Flask] No se recibió el valor de potencia en /set_power")
        return jsonify({'status': 'error', 'message': 'No se recibió el valor de potencia'}), 400

# -----------------------------
# API Endpoints para Datos de Sensores
# -----------------------------

@app.route('/sensor_data', methods=['POST'])
def sensor_data():
    try:
        data = request.get_json()
        if not data:
            logger.warning("No se recibió datos JSON en /sensor_data.")
            abort(400, description="No se recibieron datos.")

        # Validar y limpiar los datos antes de almacenarlos
        sensor_entry = {
            'timestamp': time.time(),
            'ultrasonido_1': data.get('ultrasonido_1'),
            'ultrasonido_2': data.get('ultrasonido_2'),
            'ultrasonido_3': data.get('ultrasonido_3'),
            'sensor_ultrasonico': data.get('sensor_ultrasonico'),
            'sensor_ir': data.get('sensor_ir'),
            'sensor_fuego_1': data.get('sensor_fuego_1'),
            'sensor_fuego_2': data.get('sensor_fuego_2'),
            'gps_lat': data.get('gps_lat'),
            'gps_lng': data.get('gps_lng')
        }

        db.collection('sensor_data').add(sensor_entry)
        logger.info("Datos de sensores recibidos y almacenados en Firebase.")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.exception("Error en el endpoint /sensor_data: %s", e)
        abort(500)

@app.route('/gps_data', methods=['POST'])
def gps_data():
    try:
        data = request.get_json()
        if not data:
            logger.warning("No se recibió datos JSON en /gps_data.")
            abort(400, description="No se recibieron datos.")

        sensor_entry = {
            'timestamp': time.time(),
            'gps_lat': data.get('gps_lat'),
            'gps_lng': data.get('gps_lng')
        }

        db.collection('sensor_data').add(sensor_entry)
        logger.info("Datos GPS recibidos y almacenados en Firebase.")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.exception("Error en el endpoint /gps_data: %s", e)
        abort(500)

@app.route('/get_sensor_data', methods=['GET'])
@login_required
def get_sensor_data():
    try:
        historial = Historial(db)
        sensores_data = historial.obtener_datos_sensor(limite=100)
        return jsonify(sensores_data), 200
    except Exception as e:
        logger.exception("Error al obtener datos de sensores: %s", e)
        abort(500)

# -----------------------------
# Funciones para Envío de Comandos vía MQTT
# -----------------------------

def send_mqtt_commands(recognized_faces, recognized_objects):
    # Ejemplo: Si se reconoce un rostro específico, enviar un comando
    for face in recognized_faces:
        if face == "Juan Perez":
            command = "activar_luz"
            mqtt_client.publish(MQTT_TOPIC_COMMANDS, command)
            logger.info(f"Comando MQTT enviado: {command} para {face}")

    for obj in recognized_objects:
        if obj == "Fuego":
            command = "activar_alarma"
            mqtt_client.publish(MQTT_TOPIC_COMMANDS, command)
            logger.info(f"Comando MQTT enviado: {command} para {obj}")

def process_sensor_data(payload):
    try:
        data = json.loads(payload)
        sensor_entry = {
            'timestamp': time.time(),
            'ultrasonido_1': data.get('ultrasonido_1'),
            'ultrasonido_2': data.get('ultrasonido_2'),
            'ultrasonido_3': data.get('ultrasonido_3'),
            'sensor_ultrasonico': data.get('sensor_ultrasonico'),
            'sensor_ir': data.get('sensor_ir'),
            'sensor_fuego_1': data.get('sensor_fuego_1'),
            'sensor_fuego_2': data.get('sensor_fuego_2'),
            'gps_lat': data.get('gps_lat'),
            'gps_lng': data.get('gps_lng')
        }
        db.collection('sensor_data').add(sensor_entry)
        logger.info(f"Datos de sensores recibidos vía MQTT y almacenados en Firebase: {data}")

        # Emitir a través de SocketIO para actualizaciones en tiempo real
        # Mapeo de los datos
        mapped_data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sensor_entry['timestamp'])),
            "ultrasonico_distancia": sensor_entry['sensor_ultrasonico'] if sensor_entry['sensor_ultrasonico'] is not None else "N/A",
            "ultrasonico_digital_1": "Detectado" if sensor_entry['ultrasonido_1'] == 0 else "Libre",
            "ultrasonico_digital_2": "Detectado" if sensor_entry['ultrasonido_2'] == 0 else "Libre",
            "ultrasonico_digital_3": "Detectado" if sensor_entry['ultrasonido_3'] == 0 else "Libre",
            "sensor_ir": "Detectado" if sensor_entry['sensor_ir'] == 0 else "No Detectado",
            "sensor_fuego_1": "Detectado" if sensor_entry['sensor_fuego_1'] == 0 else "No Detectado",
            "sensor_fuego_2": "Detectado" if sensor_entry['sensor_fuego_2'] == 0 else "No Detectado",
            "gps_lat": sensor_entry['gps_lat'] if sensor_entry['gps_lat'] is not None else "N/A",
            "gps_lng": sensor_entry['gps_lng'] if sensor_entry['gps_lng'] is not None else "N/A"
        }
        socketio.emit('new_sensor_data', mapped_data, broadcast=True)
    except Exception as e:
        logger.error(f"Error al procesar datos de sensores MQTT: {e}")

# -----------------------------
# Funciones de SocketIO para Actualizaciones en Tiempo Real
# -----------------------------

@socketio.on('connect')
def handle_connect_socket():
    if current_user.is_authenticated:
        logger.info(f"Cliente conectado: {current_user.username}")
        emit('connection_response', {'data': 'Conectado al servidor SocketIO'})
    else:
        logger.warning("Cliente intentó conectarse sin autenticación.")
        emit('connection_response', {'data': 'No autenticado'}, broadcast=False)
        return False  # Desconectar el cliente

@socketio.on('disconnect')
def handle_disconnect_socket():
    logger.info("Cliente desconectado.")

# -----------------------------
# Ruta para Subir y Procesar Imágenes (Opcional)
# -----------------------------

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Autenticación: verificar el token de autorización
        auth_token = request.headers.get('Authorization')
        if not auth_token or auth_token != 'Bearer tu_token_secreto_unico_aqui':
            logger.warning("Intento de acceso no autorizado al endpoint /upload.")
            abort(401, description="No autorizado")

        # Verificar que se haya enviado una imagen
        if 'image' not in request.files:
            logger.warning("No se encontró la imagen en la solicitud.")
            abort(400, description="No se encontró la imagen en la solicitud")

        image_file = request.files['image']
        if image_file.filename == '':
            logger.warning("No se seleccionó ninguna imagen.")
            abort(400, description="No se seleccionó ninguna imagen")

        # Leer los bytes de la imagen
        image_bytes = image_file.read()
        if not image_bytes:
            logger.warning("La imagen está vacía.")
            abort(400, description="La imagen está vacía")

        # Procesar la imagen usando OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("La imagen es inválida.")
            abort(400, description="La imagen es inválida")

        # Convertir a escala de grises para la detección de rostros
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_recognition.face_locations(gray)
        face_count = len(faces)

        recognized_faces = []
        recognized_objects = []

        # Reconocimiento Facial
        if face_count > 0:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Desconocido"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if face_distances.size > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                recognized_faces.append(name)
                logger.info(f"Rostro reconocido: {name}")

                # Dibujar rectángulos y nombres en la imagen
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Reconocimiento de Objetos
        recognized_objects = recognize_objects(img)

        # Codificar la imagen procesada a JPEG
        _, buffer = cv2.imencode('.jpg', img)
        processed_image_bytes = buffer.tobytes()

        # Guardar la imagen procesada como 'latest_frame.jpg'
        latest_frame_path = os.path.join(os.getcwd(), 'latest_frame.jpg')
        with open(latest_frame_path, 'wb') as f:
            f.write(processed_image_bytes)

        # Guardar la detección en Firestore
        sensor_entry = {
            'timestamp': time.time(),
            'ultrasonido_1': None,  # Asumiendo que estos campos no son relevantes aquí
            'ultrasonido_2': None,
            'ultrasonido_3': None,
            'sensor_ultrasonico': None,
            'sensor_ir': None,
            'sensor_fuego_1': None,
            'sensor_fuego_2': None,
            'gps_lat': None,
            'gps_lng': None
        }
        db.collection('sensor_data').add(sensor_entry)

        logger.info(f"Imagen procesada y guardada: {face_count} rostros detectados, {len(recognized_faces)} reconocidos, {len(recognized_objects)} objetos reconocidos.")

        # Emitir la detección a través de SocketIO para actualizaciones en tiempo real
        socketio.emit('new_detection', {
            'timestamp': sensor_entry['timestamp'],
            'face_count': face_count,
            'recognized_faces': recognized_faces,
            'recognized_objects': recognized_objects
        }, broadcast=True)

        # Enviar comandos a través de MQTT basado en detecciones
        send_mqtt_commands(recognized_faces, recognized_objects)

        return jsonify({"status": "success", "face_count": face_count, "recognized_faces": recognized_faces, "recognized_objects": recognized_objects}), 200
    except Exception as e:
        logger.exception("Error en el endpoint /upload: %s", e)
        abort(500)

# -----------------------------
# Funciones de Reconocimiento de Objetos (Opcional)
# -----------------------------

# Esta es una implementación básica. Puedes mejorarla según tus necesidades.
RECOGNIZED_OBJECTS_DIR = 'recognized_objects'
recognized_object_templates = {}

def load_recognized_objects():
    global recognized_object_templates
    if not os.path.exists(RECOGNIZED_OBJECTS_DIR):
        logger.warning(f"La carpeta {RECOGNIZED_OBJECTS_DIR} no existe.")
        return
    for filename in os.listdir(RECOGNIZED_OBJECTS_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(RECOGNIZED_OBJECTS_DIR, filename)
            template = cv2.imread(filepath, 0)  # Cargar en escala de grises
            if template is not None:
                object_name = os.path.splitext(filename)[0].replace('_', ' ').title()
                recognized_object_templates[object_name] = template
                logger.info(f"Cargado objeto reconocido: {object_name}")
            else:
                logger.warning(f"No se pudo cargar el objeto en la imagen {filename}")

load_recognized_objects()

def recognize_objects(img):
    recognized = []
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for name, template in recognized_object_templates.items():
        res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8  # Ajusta según sea necesario
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            recognized.append(name)
            logger.info(f"Objeto reconocido: {name}")
    return recognized

# -----------------------------
# Ruta para Servir la Transmisión de Video (Opcional)
# -----------------------------

@app.route('/video_feed')
@login_required
def video_feed():
    """
    Servir la transmisión de video desde el ESP32-CAM.
    Asumiendo que el ESP32-CAM está sirviendo el video en 'http://IP_DEL_ESP32_CAM/video'.
    """
    esp32_cam_url = os.getenv('ESP32_CAM_URL', 'http://192.168.1.100/video')  # Reemplaza con la IP real del ESP32-CAM
    try:
        def generate():
            with requests.get(esp32_cam_url, stream=True) as r:
                if r.status_code != 200:
                    logger.error(f"Error al conectar con ESP32-CAM: {r.status_code}")
                    abort(502, description="No se pudo conectar con la cámara")
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk
        return app.response_class(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.exception("Error al servir la transmisión de video: %s", e)
        abort(500, description="Error interno al servir la transmisión de video")

# -----------------------------
# Función Principal
# -----------------------------

if __name__ == '__main__':
    APP_PORT = int(os.getenv('APP_PORT', 5000))  # Puerto de la aplicación

    try:
        logger.info(f"Iniciando la aplicación Flask en el puerto {APP_PORT}...")
        socketio.run(app, host='0.0.0.0', port=APP_PORT)
    except Exception as e:
        logger.exception("Error al iniciar la aplicación Flask: %s", e)
        mqtt_client.disconnect()
        mqtt_client.loop_stop()
