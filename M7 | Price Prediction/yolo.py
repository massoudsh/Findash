from ultralytics import YOLO
import os
import torch
import logging
from pathlib import Path
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check if CUDA is available and return appropriate device."""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        logger.info("No GPU detected, using CPU")
    return device

def load_yolo_model(model_name="yolov8n.pt", pretrained=True):
    """
    Load a YOLO model
    
    Args:
        model_name: Name of the model to load ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')
        pretrained: Whether to load a pretrained model
        
    Returns:
        Loaded YOLO model
    """
    try:
        # Check if model file exists locally
        if not pretrained and os.path.exists(model_name):
            logger.info(f"Loading local model from {model_name}")
            model = YOLO(model_name)
        else:
            logger.info(f"Loading pretrained model {model_name}")
            model = YOLO(model_name)
            
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def train_model(model, data_yaml, epochs=100, imgsz=640, batch=16, device=None):
    """
    Train a YOLO model
    
    Args:
        model: YOLO model
        data_yaml: Path to dataset YAML
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to use for training
        
    Returns:
        Training results
    """
    if device is None:
        device = check_gpu_availability()
    
    logger.info(f"Training model on {device} for {epochs} epochs")
    
    # Training with more detailed configuration
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=50,  # Early stopping patience
        save=True,    # Save checkpoints
        project="yolo_training",
        name="model_training",
    )
    
    return results

def evaluate_model(model, data=None):
    """
    Evaluate a YOLO model
    
    Args:
        model: YOLO model
        data: Optional validation data path
        
    Returns:
        Validation metrics
    """
    logger.info("Evaluating model performance")
    metrics = model.val(data=data)
    
    # Log key metrics
    if hasattr(metrics, 'box') and hasattr(metrics.box, 'map'):
        logger.info(f"mAP50-95: {metrics.box.map:.4f}")
    if hasattr(metrics, 'box') and hasattr(metrics.box, 'map50'):
        logger.info(f"mAP50: {metrics.box.map50:.4f}")
        
    return metrics

def predict_and_visualize(model, image_path, save_dir="results", conf=0.25):
    """
    Perform prediction on an image and visualize results
    
    Args:
        model: YOLO model
        image_path: Path to image
        save_dir: Directory to save results
        conf: Confidence threshold
        
    Returns:
        Prediction results
    """
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None
        
    logger.info(f"Running prediction on {image_path}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Run prediction with visualization
    results = model.predict(
        source=image_path,
        conf=conf,
        save=True,
        project=save_dir,
        name=Path(image_path).stem
    )
    
    # Display number of detections
    if len(results) > 0:
        total_detections = sum(len(r.boxes) for r in results)
        logger.info(f"Detected {total_detections} objects")
    
    return results

def export_model(model, format="onnx", imgsz=640):
    """
    Export YOLO model to different formats
    
    Args:
        model: YOLO model
        format: Export format ('onnx', 'torchscript', 'openvino', etc.)
        imgsz: Image size for the exported model
        
    Returns:
        Path to exported model
    """
    logger.info(f"Exporting model to {format} format")
    
    try:
        path = model.export(format=format, imgsz=imgsz)
        logger.info(f"Model exported to {path}")
        return path
    except Exception as e:
        logger.error(f"Export error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Check available device
    device = check_gpu_availability()
    
    # Load YOLOv8 model (nano size)
    model = load_yolo_model("yolov8n.pt")
    
    # Example: Train the model
    # Uncomment to run training
    """
    train_results = train_model(
        model=model,
        data_yaml="coco8.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device
    )
    """
    
    # Example: Evaluate model
    # metrics = evaluate_model(model)
    
    # Example: Run prediction on a sample image
    # Replace with actual image path to test
    image_path = "path/to/image.jpg"
    if os.path.exists(image_path):
        results = predict_and_visualize(model, image_path)
    
    # Example: Export model to ONNX format
    # export_path = export_model(model, format="onnx")
    
    logger.info("YOLO operations completed")

# Deploy as an API for your fintech application
app = Flask(__name__)

@app.route('/api/analyze_chart', methods=['POST'])
def analyze_chart_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"})
    
    image = request.files['image']
    patterns = detect_chart_patterns(image)
    return jsonify({"patterns": patterns})