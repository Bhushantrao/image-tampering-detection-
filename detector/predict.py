import torch
import timm
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model at module level
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=False,
    num_classes=2
)

# Resolve absolute base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained weights using absolute paths
checkpoint_path = os.path.join(BASE_DIR, 'best_vit_model.pth')
if not os.path.exists(checkpoint_path):
    # Fallback to saved_models directory if not found in root
    checkpoint_path = os.path.join(BASE_DIR, 'saved_models', 'best_vit_model.pth')

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

new_state_dict = {}
for key, value in checkpoint.items():
    new_key = key.replace("model.", "")
    new_state_dict[new_key] = value

model.load_state_dict(
    new_state_dict,
    strict=False
)
model.to(device)
model.eval()

classes = [
    'Fake',
    'Original'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ELA Generation Helper
def convert_to_ela(path, quality=90):
    image = Image.open(path).convert("RGB")
    temp = "temp.jpg"
    image.save(temp, "JPEG", quality=quality)
    temp_image = Image.open(temp)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([e[1] for e in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Load pixels into memory and close file handles
    ela_image.load()
    image.close()
    temp_image.close()
    
    # Use clean way to handle temp file cleanup
    if os.path.exists(temp):
        try:
            os.remove(temp)
        except Exception:
            pass
    return ela_image

# ViT Reshape Transform for Grad-CAM
def reshape_transform(tensor):
    result = tensor[:, 1:, :]
    height = width = 14
    result = result.reshape(
        tensor.size(0),
        height,
        width,
        tensor.size(2)
    )
    result = result.permute(0, 3, 1, 2)
    return result

# Setup Grad-CAM
target_layers = [model.blocks[-1].norm1]
cam = GradCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform
)

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        prediction = classes[predicted.item()]
        confidence = round(probabilities[0][predicted].item() * 100, 2)
        return prediction, confidence
    except Exception as e:
        print("Prediction Error:", e)
        return "Error", 0

def predict_and_explain(image_path):
    """
    Unified pipeline that:
    1. Classifies the image as Fake or Original on the RGB image.
    2. Generates the ELA image.
    3. Runs Grad-CAM on the ELA image to localize the tampered regions.
    4. Automatically draws bounding boxes on the original image.
    5. Computes the exact tampered area percentage using dynamic thresholding.
    6. Returns prediction, confidence, paths to generated images, and percentage.
    """
    try:
        # 1. Run prediction on RGB
        image_rgb = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image_rgb.size
        input_rgb = transform(image_rgb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_rgb = model(input_rgb)
            probs = torch.softmax(out_rgb, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
        prediction = classes[pred_idx]
        confidence = round(probs[0][pred_idx].item() * 100, 2)
        
        # Make sure media directory exists in the absolute base directory
        media_dir = os.path.join(BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)
        
        # Temporary output paths (views.py will save them to Django's models)
        ela_path = os.path.join(media_dir, 'temp_ela.jpg')
        localized_path = os.path.join(media_dir, 'temp_localized.jpg')
        heatmap_path = os.path.join(media_dir, 'temp_heatmap.jpg')
        explainable_path = os.path.join(media_dir, 'temp_explainable.jpg')
        
        # 2. Convert to ELA
        ela_img = convert_to_ela(image_path)
        ela_img.save(ela_path)
        
        # 3. Grad-CAM on RGB input (targeting class 0 - Fake)
        grayscale_cam_fake = cam(input_tensor=input_rgb, targets=[ClassifierOutputTarget(0)])[0]
        
        # 4. Generate Heatmap image (overlaid on the original 224x224 RGB image and scaled to original resolution)
        image_224 = image_rgb.resize((224, 224))
        image_np = np.array(image_224).astype(np.float32) / 255.0
        visualization = show_cam_on_image(image_np, grayscale_cam_fake, use_rgb=True)
        heatmap_cv = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, heatmap_cv)
        
        # 5. Localize tampered regions using Gaussian smoothing and dynamic thresholding
        # Apply Gaussian Blur to remove isolated noise spikes and smooth gradients
        blurred_cam = cv2.GaussianBlur(grayscale_cam_fake, (9, 9), 0)
        max_activation = np.max(blurred_cam)
        threshold = max_activation * 0.60
        
        binary_mask = (blurred_cam > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        localized_img = cv2.imread(image_path)
        scale_x = orig_w / 224.0
        scale_y = orig_h / 224.0
        
        regions = []
        valid_area = 0
        total_pixels = 224 * 224
        
        # If prediction is Original, override tampered percentage to 0
        if prediction == 'Original':
            tampered_percentage = 0.0
            # Redraw original image without bounding boxes if it's Original
            localized_img = cv2.imread(image_path)
        else:
            box_id = 1
            for c in contours:
                area = cv2.contourArea(c)
                if area > 10:  # Minimum area in 224x224 resolution
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # Filter out giant background/border token leak artifacts (width/height > 65% of image size)
                    if w < 150 and h < 150:
                        orig_x = int(x * scale_x)
                        orig_y = int(y * scale_y)
                        orig_w_box = int(w * scale_x)
                        orig_h_box = int(h * scale_y)
                        
                        # Draw bounding box (Red for Fake, thickness 3)
                        cv2.rectangle(
                            localized_img,
                            (orig_x, orig_y),
                            (orig_x + orig_w_box, orig_y + orig_h_box),
                            (0, 0, 255),
                            3
                        )
                        
                        regions.append({
                            "id": box_id,
                            "x": orig_x,
                            "y": orig_y,
                            "width": orig_w_box,
                            "height": orig_h_box,
                            "area_pct": float(round((area / total_pixels) * 100, 2))
                        })
                        valid_area += area
                        box_id += 1
                        
            tampered_percentage = float(round((valid_area / total_pixels) * 100, 2))
            
        cv2.imwrite(localized_path, localized_img)
        
        # 6. Blended explainable visualization (overlay original image and Grad-CAM heatmap)
        heatmap_orig = cv2.resize(heatmap_cv, (orig_w, orig_h))
        # Overlay original image and heatmap with 50% transparency
        explainable_img = cv2.addWeighted(localized_img, 0.5, heatmap_orig, 0.5, 0)
        cv2.imwrite(explainable_path, explainable_img)
        
        return prediction, confidence, tampered_percentage, ela_path, localized_path, heatmap_path, explainable_path, regions
        
    except Exception as e:
        print("Explainable AI Pipeline Error:", e)
        # Return fallback values
        return "Error", 0, 0.0, None, None, None, None, []
