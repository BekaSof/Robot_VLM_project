import time
import cv2
import torch
import csv
import serial
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

MODEL_NAME = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
PHONE_STREAM_URL = (0)     #"http://172.20.10.13:8080/video"
SERIAL_PORT = "COM8"   # change this if needed
BAUD_RATE = 9600

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

#serial connection to arduino
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) #wait to connect
    print("Serial connection established.")
except Exception as e:
    print(f"Error connecting to serial port: {e}")
    ser = None

def send_robot_command(command: str):
    command = command.strip().upper()
    if command not in {"FORWARD", "BACK", "LEFT", "RIGHT", "STOP"}:
        command = "STOP"

    try:
        ser.write((command + "\n").encode("utf-8"))
        print(f"Sent command to Arduino: {command}")
    except Exception as e:
        print(f"Error sending command to Arduino: {e}")

#write to csv file for logging
log_file = open("vlm_log.csv", "w", newline="", encoding="utf-8")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "timestamp",
    "frame_filename",
    "ground_truth",
    "model_output",
    "command",
    "predicted_label",
    "correct",
    "latency"
])

# prompt - describes the scene in detail for navigation
prompt_text ="""
You are the perception module for a small autonomous inspection robot.
Analyse only the robot's immediate path ahead.
Ignore distant background objects and only focus on what is near the robot and relevant for movement.

Return exactly in this format:
Description: <one very short sentence>
Hazard: <none / obstacle / cable / trip hazard / person / wall / unknown>
Suggested Action: <forward / left / right / stop>

Rules:
- If the centre path is clear, choose forward.
- If the path is unsafe or uncertain, choose stop.
"""

#adding in decision logic - lighten load on the model by asking it to describe the scene
def get_command(vlm_output: str) -> str:
    text = vlm_output.lower()

    if "suggested action:" in text:
        action = text.split("suggested action:")[1].split("\n")[0].strip() # get the first word after "suggested action:"
    # check for blocked/obstacle first since thats the most important
        if "left" in action:
            return "LEFT"
        elif "right" in action:
            return "RIGHT"
        elif "forward" in action:
            return "FORWARD"
        else:
            return "STOP"  # defaults to stop in the display, safe option if unsure
    
    # fallback if format is imperfect
    if "left" in text:
        return "LEFT"
    elif "right" in text:
        return "RIGHT"
    elif "forward" in text:
        return "FORWARD"
    return "STOP"

# open webcam - in this case using the pphone webcam stream via IP, but could be easily switched to a local webcam by changing the source in VideoCapture to 0 or 1 etc depending on your system
cap = cv2.VideoCapture(PHONE_STREAM_URL)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# inference loop
inference_interval = 3.0
last_inference_time = None  # None means inference hasnt run yet
latest_output = "Waiting for first inference..."
latest_command = "STOP"
current_ground_truth = None  # for testing, we can set this to the expected 
                            #command for the current scene and see if the model gets it right, just comparing really

latencies = []
test_results = []

print("Starting webcam loop. Press 'q' or 'esc' to quit.")

while True:
    for _ in range(2):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    now = time.time()
    time_since_last = (now - last_inference_time) if last_inference_time else float("inf")

    if time_since_last >= inference_interval:
        try: #frame edits
            h, w, _ = frame.shape
            #cropping the area of the frame that gets read
            cropped = frame[int(h*0.5):h, int(w*0.2):int(w*0.8), :]  # crop to bottom half of the frame
            #save a frame every capture
            frame_filename = f"frame_{int(time.time())}.jpg"
            cv2.imwrite(frame_filename, cropped)
            # convert opencv frame to PIL for the model
            cropped = cv2.resize(cropped, (320,240)) #resize to 320x240 for the model
            image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]}
            ]

            prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            inputs = processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=40,do_sample=False)
            end_time = time.time()

            # only decode the new tokens, not the whole prompt
            new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
            output = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

            latency = end_time - start_time
            latencies.append(latency)

            command = get_command(output)
            predicted_label = "BLOCKED" if command == "STOP" else "CLEAR"

            correct = ""
            if current_ground_truth is not None: #testing mode, we have a ground truth to compare to
                correct = (predicted_label == current_ground_truth)
                test_results.append({
                    "ground_truth": current_ground_truth,
                    "predicted": predicted_label,
                    "correct": correct,
                    "latency": latency,
                    "output": output
                })
            if len(test_results) > 0:
                accuracy = sum(1 for r in test_results if r["correct"]) / len(test_results)
                print(f"Ground truth: {current_ground_truth} | Predicted: {predicted_label} | Correct: {correct}")
                print(f"Accuracy so far: {accuracy:.2%}")

            log_writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                frame_filename,
                current_ground_truth if current_ground_truth is not None else "",
                output,
                command,
                predicted_label,
                correct,
                f"{latency:.2f}"
            ])
            log_file.flush()

            latest_command = command
            latest_output = output
            last_inference_time = now

            print("Model response:")
            print(output)
            print(f"Command: {command}")
            print(f"Inference time: {latency:.2f} seconds")

            if len(latencies) > 0:
                avg_latency = sum(latencies) / len(latencies)  
                print(f"Average Latency so far: {avg_latency:.2f} seconds")
                send_robot_command(command)

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

    cv2.putText(frame, f"Command: {latest_command}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 55 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


    cv2.imshow("Webcam VLM", frame)
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):  # 'c' to set current ground truth for testing as CLEAR to go forward
        current_ground_truth = "CLEAR"
        print("Truth set to CLEAR")
    elif key == ord('b'):
        current_ground_truth = "BLOCKED"
        print("Truth set to BLOCKED")
    elif key == ord("n"):
        current_ground_truth = None
        print("Ground truth cleared")
    elif key == ord('q') or key == 27:  # 'q' or ESC to quit
        print("Quitting...")
        break

cap.release()
log_file.close()
ser.close()
cv2.destroyAllWindows()