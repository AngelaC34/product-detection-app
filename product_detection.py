import streamlit as st # for building web app
from PIL import Image # for opening and processing images
import torch # PyTorch core library
from torchvision import transforms # for image preprocessing (ressize, to tensor, normalize)
import timm # PyTorch Image Models library ( for ViT model )
import cv2 # OpenCV for image processing and HSV detection
import numpy as np # for numerical operations

# CONFIGURATION
# Mapping product names to class index
CLASS_MAP = {
    'Chiki Twist': 0,
    'Chitato Lite Aburi Seaweed': 1,
    'Chitato Chijeu': 2,
    'Chitato Lite Kimchi': 3,
    'Chitato Lite Rumput Laut': 4,
    'Chitato Lite Salmon Teriyaki': 5,
    'Chitato Lite Saus Krim Bawang': 6,
    'Chitato Original': 7,
    'Chitato Sapi Bumbu Bakar': 8,
    'Chitato Sapi Panggang': 9,
    'Chitato Tteokbokki': 10,
    'French Fries 2000': 11
}

# reverse map index to class map
IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}

# Map class name to image path
CLASS_IMAGES = {
    "Chiki Twist": "assets/chiki_twist.png",
    "Chitato Lite Aburi Seaweed": "assets/chitato_lite_aburi_seaweed.png",
    "Chitato Chijeu": "assets/chitato_chijeu.png",
    "Chitato Lite Kimchi": "assets/chitato_lite_kimchi.png",
    "Chitato Lite Rumput Laut": "assets/chitato_lite_rumput_laut.png",
    "Chitato Lite Salmon Teriyaki": "assets/chitato_lite_salmon_teriyaki.png",
    "Chitato Lite Saus Krim Bawang": "assets/chitato_lite_saus_krim_bawang.png",
    "Chitato Original": "assets/chitato_original.png",
    "Chitato Sapi Bumbu Bakar": "assets/chitato_sapi_bumbu_bakar.png",
    "Chitato Sapi Panggang": "assets/chitato_sapi_panggang.png",
    "Chitato Tteokbokki": "assets/chitato_tteokbokki.png",
    "French Fries 2000": "assets/french_fries_2000.png",
}

# Prediction confidence threshold
THRESHOLD = 0.000001
SCALE_FACTOR = 1.3

# Model path
MODEL_PATH = "vit_base_patch16_224_bs16_ep50.pth"

# Use GPU if available, else use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default HSV threshold for skin detection
DEFAULT_H = (0, 20)
DEFAULT_S = (10, 150)
DEFAULT_V = (60, 255)


# LOAD MODEL
# Load ViT model and apply trained weights
@st.cache_resource
def load_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(CLASS_MAP)) # create ViT model
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE) # load trained weights
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval() # set model to evaluation mode
    return model

# IMAGE PREPROCESSING
@st.cache_resource
def get_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)), # Resize 224x224
        transforms.ToTensor(), # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
    ])

# SKIN DETECTION AND CROPPING
def detect_hand_hsv(pil_image, min_area=0, highlight=False):
    # Detect skin in image and return a cropped PIL image
    # Convert PIL to OpenCV
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask using HSV thresholds to detect skin
    lower_skin = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_skin = np.array([h_max, s_max, v_max], dtype=np.uint8)   

    mask = cv2.inRange(img_hsv, lower_skin, upper_skin)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Default crop parameters (center crop)
    img_h, img_w, _ = img.shape
    cx, cy = img_w // 2, img_h // 2
    crop_size = min(img_w, img_h) // 2

    # If a skin region is detected, center crop around largest contour
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > min_area:
            # Optional: highlight detected hand
            if highlight:
                cv2.drawContours(img_rgb, [largest], -1, (255, 255, 255), thickness=cv2.FILLED)

            # bounding box of detected hand
            x, y, w, h = cv2.boundingRect(largest)
            cx, cy = x + w // 2, y + h // 2

            # make square size 30% bigger than the bounding box max dimension
            crop_size = int(max(w, h) * SCALE_FACTOR)

    # Ensure crop is within image bounds
    crop_size = min(crop_size, min(img_w, img_h))
    half = crop_size // 2
    x_min = np.clip(cx - half, 0, img_w - crop_size)
    y_min = np.clip(cy - half, 0, img_h - crop_size)
    x_max, y_max = x_min + crop_size, y_min + crop_size

    cropped = img_rgb[y_min:y_max, x_min:x_max]

    return Image.fromarray(cropped)

def predict_product(model, preprocess, image):
    cropped_img = detect_hand_hsv(image) # crop around detected hand
    input_tensor = preprocess(cropped_img).unsqueeze(0).to(DEVICE) # preprocess and add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0] # get probabilities

    # Filter predictions by threshold and sort by confidence
    predictions = [
        (IDX_TO_CLASS[i], float(prob))
        for i, prob in enumerate(probs) if prob > THRESHOLD
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return cropped_img, predictions

# PRODUCT DETECTION APP
model = load_model()
preprocess = get_preprocess()

col1, col2 = st.columns([1, 1], gap="large", width="stretch")

# INPUT COLUMN
with col1:
    st.header("🖼️ Input")

    method = st.radio("Chooose input method:", ["Upload", "Capture"], horizontal=True)

    # Image Input
    image = None
    if method == "Upload":
        image = st.file_uploader("Upload an Image 📤", type=["jpg", "jpeg", "png"])
    elif method == "Capture":
        image = st.camera_input("Capture an Image 📷")

    # Start Detection Button
    if image is not None:
        with st.container(horizontal_alignment="center"):
            st.caption("Uploaded Image", width="content")
            st.image(image, width=500)

        # Initialize HSV session state if not set
        if 'hsv_values' not in st.session_state:
            st.session_state['hsv_values'] = {
                'h': list(DEFAULT_H),
                's': list(DEFAULT_S),
                'v': list(DEFAULT_V)
            }

        with st.expander("Adjust HSV Thresholds", expanded=False):
            h_min, h_max = st.select_slider(
                "Hue range",
                options=list(range(0, 180)),
                value=tuple(st.session_state['hsv_values']['h'])
            )
            s_min, s_max = st.select_slider(
                "Saturation range",
                options=list(range(0, 256)),
                value=tuple(st.session_state['hsv_values']['s'])
            )
            v_min, v_max = st.select_slider(
                "Value range",
                options=list(range(0, 256)),
                value=tuple(st.session_state['hsv_values']['v'])
            )

            # Update session state when slider values change
            st.session_state['hsv_values']['h'] = [h_min, h_max]
            st.session_state['hsv_values']['s'] = [s_min, s_max]
            st.session_state['hsv_values']['v'] = [v_min, v_max]


        if st.button("Start Detection", type="primary", width="stretch"):
            img = Image.open(image).convert("RGB")
            cropped_img, predictions = predict_product(model, preprocess, img)
            st.session_state['result'] = (cropped_img, predictions)
    
# RESULT COLUMN
with col2:
    st.header("🎯 Result")
    with st.container(horizontal_alignment="center"):
        if 'result' in st.session_state:
            img, predictions = st.session_state['result']
            st.caption("Cropped Image", width="content")
            st.image(img, width=500)

            if predictions:
                st.write("**Detected Product(s):**")
                # Display top 3 predictions
                for cls, conf in predictions[:3]:
                    image_path = CLASS_IMAGES.get(cls)

                    res1, res2 = st.columns([1, 4], vertical_alignment="center")
                    if image_path:
                        res1.image(image_path, width=100)
                    res2.markdown(f"**{cls}**")

        else:
            st.image("assets/placeholder.png", caption="No prediction yet")