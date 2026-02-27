# AI Coding Agent Instructions

This repository is a small Python project for a third-year autonomous robot
using a Visual Language Model (VLM). The codebase is minimal today, but the
pipeline is outlined in `README.md` and will be expanded over the course of the
project.

## 📦 Project Overview

* **Pipeline:** `Camera -> OpenCV -> VLM -> decision -> (serial) -> Arduino -> motors`
  - The README describes the overall flow; any new code should align with this
    sequence.
* **Current state:** `src/main.py` contains only a basic OpenCV camera demo that
  opens the default webcam, displays frames in a window, and exits on `q`.

## 🗂 Key Files & Structure

* `src/main.py` – entry point. Reads frames from `cv2.VideoCapture` and shows
  them. Resource management (opening/closing camera) is handled here.
* `README.md` – high‑level description of the robot’s architecture and pipeline.
* No other modules or tests exist yet; feel free to add new Python modules
  within `src/` as features are developed.

## ⚙️ Developer Workflows

1. **Setup dependencies** (assumes a virtualenv):
   ```sh
   pip install opencv-python pyserial
   # plus any VLM-specific packages later (e.g. torch, transformers, etc.)
   ```
2. **Run the demo:**
   ```sh
   python src/main.py
   ```
   - A webcam must be attached and accessible as device `0`.
   - Press `q` in the display window to quit cleanly.
3. **Testing & debugging:**
   - There are no automated tests yet; when adding logic, include `pytest`‑style
     tests under a new `tests/` directory.
   - Use simple `print()` statements or the `logging` module as needed.

## 🧩 Conventions & Patterns

* Keep functions small and focused. If you add a new responsibility (e.g., VLM
  inference, decision making, serial communication), place it in its own
  module and import it from `main.py`.
* Follow Python idioms (snake_case, explicit imports, context managers where
  appropriate). The code should remain readable for undergraduate developers.
* When interacting with hardware:
  * Check `cap.isOpened()` after creating a `VideoCapture`.
  * Verify `ret` each frame before processing.
  * Release resources with `cap.release()` and `cv2.destroyAllWindows()`.
  * For Arduino communication use `pyserial`, open the port once, send
    formatted commands (e.g., ASCII strings), and close the port on shutdown.

## 🔗 Integration Points

* **VLM module:** Accepts frames (NumPy arrays) and returns semantic labels or
  predictions. Can be implemented with any library (PyTorch, TensorFlow, etc.).
* **Decision logic:** Converts VLM output into motor commands (e.g., forward,
  turn). Keep this decoupled from the camera layer to facilitate testing.
* **Serial communication:** The Arduino interface is over serial. Commands are
  sent as text and interpreted by the Arduino sketch running on the robot.

## 💡 Tips for AI Agents

* Always read `README.md` first for context.
* Modify existing code only when necessary; prefer adding new modules.
* Write concise comments explaining "why" decisions are made, not just "what"
  code does.
* If you suggest new project structure (tests, configs), explain the rationale
  based on the current minimal layout.
* There are no hidden build scripts or CI; tasks are manual and simple.

---

*After adding or changing logic, ask the user for feedback if the instructions
are still relevant or if additional details are needed.*
