from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np


CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="UguVp8wiRQ1WIc8Ihzo0"
)

def estimate_damage(image_path, model_id="etiquetado-de-danos/1", output_path="output.jpg"):

    image = cv2.imread(image_path)
    height, width, _ = image.shape
    total_area = height * width

    result = CLIENT.infer(image_path, model_id=model_id)
    

    damage_area = 0
    for prediction in result["predictions"]:
        x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        bbox_area = w * h
        damage_area += bbox_area
        

        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f"Damage", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    # cv2.imshow("Damage Estimation", image)
    # cv2.waitKey(0)
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")
    
    damage_percentage = (damage_area / total_area) * 100
    cv2.imshow("Damage Estimation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Total Damage Area: {damage_area} pixels")
    print(f"Damage Percentage: {damage_percentage:.2f}% of the image")
    # print("Detections:")
    # for detection in result["predictions"]:
    #     print(f" - {detection['label']}: {detection['confidence']:.2f}")
    
    return {
        "total_damage_area": damage_area,
        "damage_percentage": damage_percentage,
        "detections": result["predictions"]
    }


image_path = r"C:\Users\nithi\Downloads\istockphoto-172181182-612x612.jpg" 

output_path = "visualized_output.jpg"
output = estimate_damage(image_path, output_path=output_path)
print(output)
