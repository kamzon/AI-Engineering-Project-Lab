import os
import requests
import random
import io
import base64
from PIL import Image, ImageFilter
import numpy as np

AI_ENDPOINT = "https://llm-web.aieng.fim.uni-passau.de/v1/images/generations"
API_UPLOAD_ENDPOINT = "http://localhost:8000/api/count/"
API_CORRECT_ENDPOINT = "http://localhost:8000/api/correct/"

API_KEY ="gpustack_45a2e6ef3e10605a_2610a6c7464852e488494e2bd605d1bb"

OBJECT_TYPES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "building", "road", "sky", "ground", "water"
]
BACKGROUND_TYPES = ["random", "solid", "noise"]
NUM_IMAGES = 10
OBJECTS_PER_IMAGE = (1, 3)

def generate_image_with_api(prompt, size="512x512", model="flux.1-schnell-gguf", api_key=None):
    url = AI_ENDPOINT
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "n": 1,
        "size": size,
        "seed": None,
        "sample_method": "euler",
        "cfg_scale": 1,
        "guidance": 3.5,
        "sampling_steps": 20,
        "negative_prompt": "",
        "strength": 0.75,
        "schedule_method": "discrete",
        "model": model,
        "prompt": prompt
    }
    response = requests.post(url, headers=headers, json=data)
    if response.ok:
        b64_img = response.json()['data'][0]['b64_json']
        img_bytes = base64.b64decode(b64_img)
        img = Image.open(io.BytesIO(img_bytes))
        return img
    else:
        print("Image generation failed:", response.text)
        return None

def augment_image(img, blur=0, rotate=0, noise=0):
    if img is None:
        return None
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    if rotate != 0:
        img = img.rotate(rotate)
    if noise > 0:
        arr = np.array(img)
        noise_arr = np.random.normal(0, noise, arr.shape).astype(np.uint8)
        arr = np.clip(arr + noise_arr, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
    return img

def post_image_to_api(img, object_type, correct_count):
    if img is None:
        print("No image to upload.")
        return
    save_dir = "generated_images"
    os.makedirs(save_dir, exist_ok=True)
    img_filename = f"{object_type}_{random.randint(100000,999999)}.png"
    img_path = os.path.join(save_dir, img_filename)
    img.save(img_path)
    print(f"Saved generated image to {img_path}")

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    files = {'image': ('test.png', buf, 'image/png')}
    data = {'object_type': object_type}
    response = requests.post(API_UPLOAD_ENDPOINT, files=files, data=data)
    if response.ok:
        result = response.json()
        correction_data = {
            "result_id": result["id"],
            "corrected_count": correct_count
        }
        requests.post(API_CORRECT_ENDPOINT, data=correction_data)
        print(f"Tested image for {object_type} with count {correct_count}")
    else:
        print("API upload failed:", response.text)

if __name__ == "__main__":
    for i in range(NUM_IMAGES):
        num_objects = random.randint(*OBJECTS_PER_IMAGE)
        chosen_types = random.choices(OBJECT_TYPES, k=num_objects)
        background = random.choice(BACKGROUND_TYPES)
        blur = 0
        rotate = random.choice([0, 90, 180, 270])
        noise = 0

        prompt = f"A {background} background with " + ", ".join(chosen_types)

        try:
            img = generate_image_with_api(prompt, api_key=API_KEY)
        except Exception as e:
            print(e)
            continue

        img = augment_image(img, blur=blur, rotate=rotate, noise=noise)

        selected_object = chosen_types[0]
        correct_count = chosen_types.count(selected_object)
        post_image_to_api(img, selected_object, correct_count=correct_count)