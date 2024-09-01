import os
import torch
from PIL import Image
import gradio as gr
from transformers import DetrImageProcessor, DetrForObjectDetection
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve Hugging Face token from environment variable
HF_TOKEN = os.getenv('HF_TOKEN')

if HF_TOKEN is None:
    raise ValueError("Hugging Face token not found in environment variables.")

# Login to Hugging Face using the token
login(token=HF_TOKEN)

# Load DETR model for object detection
def load_detr_model():
    try:
        model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
        processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
        return model, processor, None
    except Exception as e:
        return None, None, f"Error loading DETR model: {str(e)}"

detr_model, detr_processor, detr_error = load_detr_model()

def detect_objects(image):
    if image is None:
        return None, "Invalid image: image is None."

    if detr_model is not None and detr_processor is not None:
        try:
            inputs = detr_processor(images=image, return_tensors="pt")
            outputs = detr_model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            detected_objects = [
                {"label": detr_model.config.id2label[label.item()],
                 "box": box.tolist()}
                for label, box in zip(results['labels'], results['boxes'])
            ]
            return detected_objects, None
        except Exception as e:
            return None, f"Error in detect_objects: {str(e)}"
    else:
        return None, "DETR models not loaded. Skipping object detection."

# Load Stable Diffusion model for image generation
def load_stable_diffusion_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
        return pipeline, None
    except Exception as e:
        return None, f"Error loading Stable Diffusion model: {str(e)}"

sd_pipeline, sd_error = load_stable_diffusion_model()

def adjust_dimensions(width, height):
    # Adjust width and height to be divisible by 8
    adjusted_width = (width // 8) * 8
    adjusted_height = (height // 8) * 8
    return adjusted_width, adjusted_height

def generate_image(prompt, width, height):
    if sd_pipeline is not None:
        try:
            adjusted_width, adjusted_height = adjust_dimensions(width, height)
            image = sd_pipeline(prompt, width=adjusted_width, height=adjusted_height).images[0]
            # Resize back to original dimensions if needed
            image = image.resize((width, height), Image.LANCZOS)
            return image, None
        except Exception as e:
            return None, f"Error in generate_image: {str(e)}"
    else:
        return None, "Stable Diffusion model not loaded. Skipping image generation."

def process_image(image):
    if image is None:
        return None, "Invalid image: image is None."

    try:
        # Detect objects in the provided image
        detected_objects, detect_error = detect_objects(image)
        if detect_error:
            return None, detect_error

        # Create a prompt based on detected objects
        prompt = "modern redesign of an interior room with "
        if detected_objects:
            prompt += ", ".join([obj['label'] for obj in detected_objects])
        else:
            prompt += "empty room"

        # Generate a redesigned image based on the prompt
        width, height = image.size
        generated_image, gen_image_error = generate_image(prompt, width, height)
        if gen_image_error:
            return None, gen_image_error

        return generated_image, None
    except Exception as e:
        return None, f"Error in process_image: {str(e)}"

# Custom CSS for styling
custom_css = """
body {
    background-color: black;
}

h1 {
    background: linear-gradient(to right, blue, purple);
    -webkit-background-clip: text;
    color: transparent;
    font-size: 3em;
    text-align: center;
    margin-bottom: 20px;
}
"""

# Creating the Gradio interface with custom styling
iface = gr.Interface(
    fn=process_image,
    inputs=[gr.Image(type="pil", label="Upload Room Image")],
    outputs=[gr.Image(type="pil", label="Redesigned Image"), gr.Textbox(label="Error Message")],
    title="Interior Redesign",
    css=custom_css
)

try:
    iface.launch()
except Exception as e:
    print(f"Error occurred while launching the interface: {str(e)}")
