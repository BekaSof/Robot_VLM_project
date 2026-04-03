# Import required libraries and set hardware for model to use
import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

#choose the device to run the model on
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

#load the image(frame)
image = Image.open("frame.jpg").convert("RGB")

#load the processor
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

#load the model and move it to the device
print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME)
model = model.to(device)

#Creating a prompt for the model to generate a response
messages = [
    { "role": "user" , "content": [
            {"type": "image"},
            {"type": "text", "text":"Analyse this image for a robot.\n"
                                    "Describe the scene briefly and decide an action.\n"
                                    "Only output:\n"
                                    "VLM Output: <scene description>\n"
                                     "Action: <forward, left, right, stop>"} ]
    }
]

#input processing
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

inputs = processor(
    text=prompt,
    images=[image],
    return_tensors="pt"
)
#moves it to the device
inputs = {k: v.to(device) for k, v in inputs.items()}

#timing
start_time = time.time()
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=40)
end_time = time.time()

#decoding the response from the model

output = processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
print("Model response:")
print(output)
print(f"Time taken: {end_time - start_time:.2f} seconds")