# app.py
import os
from flask import Flask, request, render_template, send_from_directory, jsonify
import cv2
import numpy as np
import logging # Import logging module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask application
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logging.info(f"Created UPLOAD_FOLDER: {UPLOAD_FOLDER}")
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)
    logging.info(f"Created PROCESSED_FOLDER: {PROCESSED_FOLDER}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size: 16MB

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- YOLO Model Functions ---

# This function loads the YOLO model from model.pt using ultralytics
def load_yolo_model():
    """
    Loads the YOLO model from 'model.pt' using ultralytics.YOLO.
    If ultralytics is not installed or model.pt cannot be loaded,
    it falls back to a mock setup.
    """
    logging.info("Attempting to load YOLO model from model.pt using ultralytics...")
    model_instance = None
    classes = []

    try:
        from ultralytics import YOLO
        # This will load your 'model.pt' file. Make sure model.pt is in the same directory as app.py
        model_instance = YOLO("model.pt")
        classes = model_instance.names # ultralytics models have a .names attribute for class names
        logging.info("YOLO model loaded successfully from model.pt using ultralytics.")
        return model_instance, classes, None # None for output_layers, as it's not used by ultralytics models
    except ImportError:
        logging.warning("ultralytics library not found. Please install it (pip install ultralytics) to use real YOLO models.")
    except FileNotFoundError:
        logging.error("model.pt not found. Ensure 'model.pt' is in the same directory as app.py.")
    except Exception as e:
        logging.error(f"Error loading model.pt with ultralytics: {e}", exc_info=True)
        logging.warning("Falling back to mock YOLO.")

    # Fallback to mock model if real loading fails
    logging.info("Using mock YOLO model.")
    return "mock_yolo_net", ["person", "car", "bicycle"], ["output_layer_1"]


# Load the YOLO model once when the app starts
yolo_net, yolo_classes, yolo_output_layers = load_yolo_model()
logging.info(f"YOLO classes loaded: {yolo_classes}")


# This function performs object detection using the loaded model.
def detect_objects(image_path, model, classes, output_layers):
    """
    Performs object detection using the actual YOLO model or simulates it.
    """
    logging.info(f"Performing object detection for {image_path}...")
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}. Check file path and permissions.")

        if model in ["mock_yolo_net"]: # Check if it's the mock model fallback
            # Existing mock detection logic
            height, width, channels = img.shape
            mock_detections = []
            if not classes or not isinstance(classes, list) or len(classes) == 0:
                logging.error("YOLO classes not properly loaded for mock detection.")
                classes = ["object"] # Fallback

            if len(classes) > 0:
                if np.random.rand() > 0.3:
                    person_class_id = classes.index("person") if "person" in classes else 0
                    x1, y1, w1, h1 = int(width * 0.2), int(height * 0.3), int(width * 0.3), int(height * 0.6)
                    mock_detections.append([x1, y1, w1, h1, 0.95, person_class_id])

                if np.random.rand() > 0.5:
                    car_class_id = classes.index("car") if "car" in classes else 1
                    x2, y2, w2, h2 = int(width * 0.6), int(height * 0.5), int(width * 0.3), int(height * 0.4)
                    mock_detections.append([x2, y2, w2, h2, 0.88, car_class_id])
                
                if np.random.rand() > 0.7:
                    random_class_id = np.random.randint(0, len(classes))
                    x3, y3, w3, h3 = int(width * 0.1), int(height * 0.1), int(width * 0.2), int(height * 0.2)
                    mock_detections.append([x3, y3, w3, h3, 0.75, random_class_id])

            boxes = []
            confidences = []
            class_ids = []

            for detection in mock_detections:
                x, y, w, h, confidence, class_id = detection


                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

            indexes = []
            if boxes:
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                if isinstance(indexes, tuple):
                    indexes = []
                elif isinstance(indexes, np.ndarray):
                    indexes = indexes.flatten()
                else:
                    indexes = []

            final_detections = []
            if len(indexes) > 0:
                for i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]]) if class_ids[i] < len(classes) else "unknown"
                    confidence = confidences[i]
                    final_detections.append({
                        "box": [x, y, w, h],
                        "label": label,
                        "confidence": round(confidence, 2)
                    })
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            logging.info(f"Mock detection completed. Detections found: {len(final_detections)}")
            return img, final_detections
        else:
            # --- Real YOLO Model Inference (using ultralytics) ---
            logging.info("Performing real YOLO inference.")
            
            if hasattr(model, 'predict'):
                results = model.predict(img, verbose=False) # 'img' is a numpy array (OpenCV image)
                
                final_detections = []
                for r in results:
                    # r.boxes contains detected bounding boxes, scores, and class IDs
                    for box in r.boxes:
                        # Extract bounding box coordinates (xyxy format: top-left x,y, bottom-right x,y)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = round(float(box.conf[0]), 2)

                        if(conf < 0.9): continue
                        else:
                            cls = int(box.cls[0])
                            
                            label = classes[cls] if cls < len(classes) else "unknown"

                            # Draw bounding box on the image
                            color = (0, 255, 0)  # Green color
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            # Store detection in a format for JSON response
                            final_detections.append({
                                "box": [x1, y1, x2 - x1, y2 - y1], # Convert to x, y, w, h
                                "label": label,
                                "confidence": conf
                            })
                logging.info(f"Real YOLO detection completed. Detections found: {len(final_detections)}")
                return img, final_detections
            else:
                logging.error("Loaded model does not have a 'predict' method or is not a recognized YOLO object. Falling back to mock detection logic.")
                # Fallback to mock logic if the loaded model doesn't behave as expected
                return detect_objects(image_path, "mock_yolo_net", classes, output_layers) # Recursive call to mock path

    except Exception as e:
        logging.error(f"Error during object detection: {e}", exc_info=True)
        img_original = cv2.imread(image_path)
        if img_original is None:
            img_original = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(img_original, "Error: Could not load image!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
             cv2.putText(img_original, f"Processing Error: {str(e)[:50]}...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img_original, []


# --- Flask Routes ---

@app.route('/')
def index():
    """
    Renders the main page where users can upload images.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles the image upload, processes it with the mock YOLO model,
    and returns the path to the processed image.
    Ensures all responses are JSON, even on error.
    """
    try:
        if 'file' not in request.files:
            logging.error("No file part in request.")
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            logging.error("No selected file name.")
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            logging.info(f"File saved to: {filepath}")

            processed_img, detections = detect_objects(filepath, yolo_net, yolo_classes, yolo_output_layers)

            processed_filename = "processed_" + filename
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            
            os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
            cv2.imwrite(processed_filepath, processed_img)
            logging.info(f"Processed image saved to: {processed_filepath}")

            return jsonify({
                "processed_image_url": f"/processed/{processed_filename}",
                "detections": detections
            })
        else:
            logging.warning(f"Invalid file type uploaded: {file.filename}")
            return jsonify({"error": "Allowed image types are png, jpg, jpeg, gif"}), 400
    except Exception as e:
        logging.error(f"An unhandled error occurred in upload_file: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/processed/<filename>')
def uploaded_file(filename):
    """
    Serves the processed image from the processed folder.
    """
    logging.info(f"Serving processed file: {filename}")
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

