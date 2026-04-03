import time
import cv2
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

#setting up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#processor
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

#model
print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, dtype=torch.float16 if device == "cuda" else torch.float32)
model = model.to(device)

model.eval()
print("Model loaded succesfully.")

#prompt
prompt_text = ("You are the vision system for a small hazard-detection robot.\n"
    "Look only at the main area in front of the robot.\n"
    "Identify the main hazard and choose one robot action.\n"
    "Possible hazards: NONE, OBSTACLE, SPILL, PERSON, UNKNOWN.\n"
    "Possible actions: FORWARD, LEFT, RIGHT, STOP.\n")

#open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

#inference loop
inference_num = 3.0
inference_times = 0
newest_output = "Waiting for first inference..."

print("Starting webcam loop. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    display_frame = frame.copy()
    now = time.time()

    if now - inference_times >= inference_num:
        try:   #changing the openCV image to PIL format and processing it with the model
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            messages = [
                { "role": "user" , "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text} ]
                }
            ]

            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
            end_time = time.time()

            output = processor.batch_decode(generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            newest_output = output
            print("Model response:")
            print(output)
            print(f"Inference time: {end_time - start_time:.2f} seconds")
            inference_times = now

        except Exception as e:
            newest_output = f"Error:{str(e)}"
            print(newest_output)
            inference_times = now

    #text on screen
    cv2.putText(display_frame, newest_output, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Webcam VLM", display_frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' or ESC to quit the program!
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
