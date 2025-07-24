from flask import Flask, render_template, request, redirect, url_for, send_file
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="UguVp8wiRQ1WIc8Ihzo0"
)
def calculate_cost(damage_area, damage_percentage):
    cost = 0
    
    if damage_area > 50000:  # Large area damage
        if damage_percentage < 10:
            cost += 15000
        elif damage_percentage < 30:
            cost += 30000
        else:
            cost += 50000
    elif damage_area > 10000:  # Medium area damage
        if damage_percentage < 10:
            cost += 5000
        elif damage_percentage < 30:
            cost += 10000
        else:
            cost += 20000
    else:  # Small area damage
        if damage_percentage < 10:
            cost += 1500
        elif damage_percentage < 30:
            cost += 3000
        else:
            cost += 5000
            
    return cost

def estimate_damage(image_path, output_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    total_area = height * width
    
    result = CLIENT.infer(image_path, model_id="etiquetado-de-danos/1")
    
    damage_area = 0
    for prediction in result["predictions"]:
        x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        bbox_area = w * h
        damage_area += bbox_area
        
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, "Damage", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    damage_percentage = (damage_area / total_area) * 100
    cv2.imwrite(output_path, image)
    
    return {
        "total_damage_area": damage_area,
        "damage_percentage": damage_percentage,
        "detections": result["predictions"]
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("damage.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(url_for('index'))
    
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for('index'))
    
    if file:
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"processed_{file.filename}")
        file.save(image_path)
        
        result = estimate_damage(image_path, output_path)
        cost = calculate_cost(result['total_damage_area'], result['damage_percentage'])
        
        return render_template("damage.html",
                             image_path=f"/static/uploads/{file.filename}",
                             processed_image=f"/static/outputs/processed_{file.filename}",
                             damage_area=f"{result['total_damage_area']:.2f}",
                             damage_percentage=f"{result['damage_percentage']:.2f}",
                             cost = cost)

@app.route("/download")
def download():
    return send_file(os.path.join(OUTPUT_FOLDER, "output.jpg"), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
