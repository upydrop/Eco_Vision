import tensorflow as tf
import numpy as np
from PIL import Image
import json

MODEL_PATH = "outputs/waste_classifier.tflite"
IMAGE_PATH = "plastic_test.jpg"
LABELS_PATH = "outputs/labels.json"

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH,
    num_threads=4
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]["shape"])
print("Input dtype:", input_details[0]["dtype"])
print("Output shape:", output_details[0]["shape"])

image = Image.open(IMAGE_PATH).convert("RGB")

# Same resize as WasteClassifier.kt; the model normalizes 0..255 itself.
image = image.resize((224, 224), Image.Resampling.BILINEAR)

input_data = np.array(image, dtype=np.float32)
input_data = np.expand_dims(input_data, axis=0)

if tuple(input_data.shape) != tuple(input_details[0]["shape"]):
    raise ValueError("Image shape does not match model input")

if input_details[0]["dtype"] != np.float32:
    raise ValueError("Model input is not Float32")

interpreter.set_tensor(
    input_details[0]["index"],
    input_data
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_details[0]["index"]
)[0]

print("\n========== WASTE CLASSIFICATION ==========")

for i in np.argsort(output)[::-1]:
    print(f"{labels[i]}: {output[i]:.4f}")

best_index = np.argmax(output)

print("\n========== FINAL RESULT ==========")
print("Class:", labels[best_index])
print("Confidence:", f"{output[best_index] * 100:.2f}%")
