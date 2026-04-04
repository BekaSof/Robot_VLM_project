import time
import cv2
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

MODEL_NAME = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# setting up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Using device: {device}")

# processor
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# model
print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
model.eval()
print("Model loaded successfully.")

# prompt - describes the scene in detail for navigation
prompt_text = (
    "You are assisting a small indoor robot. "
    "Describe only what is directly in front of the robot that affects its movement. "
    "Say whether the path ahead looks clear or blocked, what (if anything) is in the way, "
    "and whether it seems safe to move forward. "
    "Keep the answer short and focused."
)

#adding in decision logic - lighten load on the model by asking it to only describe the scene and then we will do the action decision logic ourselves based on keywords in the description
def get_command(vlm_output):
    text = vlm_output.lower()

    # check for blocked/obstacle first since thats the most important
    if any(word in text for word in ["blocked", "obstacle", "wall", "stop", "cannot", "blocking", "obstructed"]):
        return "STOP"
    elif any(word in text for word in ["turn left", "go left", "left side", "left is clear"]):
        return "LEFT"
    elif any(word in text for word in ["turn right", "go right", "right side", "right is clear"]):
        return "RIGHT"
    elif any(word in text for word in ["clear", "free", "open", "forward", "proceed", "nothing blocking"]):
        return "FORWARD"
    else:
        return "STOP"  # defaults to stop in the display, safe option if unsure

# open webcam
cap = cv2.VideoCapture("http://172.20.10.12:8080/video")

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# inference loop
inference_interval = 3.0
last_inference_time = None  # None means inference hasnt run yet
latest_output = "Waiting for first inference..."
latest_command = "Stop"

print("Starting webcam loop. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    now = time.time()
    time_since_last = (now - last_inference_time) if last_inference_time else float("inf")

    if time_since_last >= inference_interval:
        try:
            # convert opencv frame to PIL for the model
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]}
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
                generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            end_time = time.time()

            # only decode the new tokens, not the whole prompt
            new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
            output = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

            command = get_command(output)
            latest_command = command
            latest_output = output
            last_inference_time = now

            print("Model response:")
            print(output)
            print(f"Command: {command}")
            print(f"Inference time: {end_time - start_time:.2f} seconds")

        except Exception as e:
            latest_output = f"Error: {str(e)}"
            last_inference_time = now
            print(latest_output)

    # wrap text across multiple lines so it doesnt go off screen
    words = latest_output.split()
    lines, current_line = [], ""
    for word in words:
        test = f"{current_line} {word}".strip()
        (w, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        if w > frame.shape[1] - 20 and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test
    if current_line:
        lines.append(current_line)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.imshow("Webcam VLM", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' or ESC to quit
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()