import numpy as np
import gradio as gr
from PIL import Image
from tensorflow import keras

MODEL_PATH = "flower_transfer_learning_mobilenetv2.keras"

CLASS_NAMES = [
    'daisy', 
    'dandelion', 
    'roses', 
    'sunflowers', 
    'tulips'
]

IMAGE_SIZE = (160, 160)

model = keras.models.load_model(MODEL_PATH)

print("Modellen är laddad.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(image: Image.Image) -> dict:
    if image is None:
        return {}
    
    image_array = prepare_image(image)

    probabilities = model.predict(image_array, verbose=0)[0]

    results = {
        class_name: float(probability)
        for class_name, probability in zip(CLASS_NAMES, probabilities)
    }

    return results


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Ladda upp en bild"),
    outputs=gr.Label(num_top_classes=5, label="Modellens prediktion"),
    title="Blomklassificering med transfer learning",
    description=(
        "Ladda upp en bild på en blomma. "
        "Modellen försöker klassificera bilden som en av fem blomklasser: "
        "daisy, dandelion, roses, sunflower eller tulips"
    )
)

if __name__ == "__main__":
    demo.launch()