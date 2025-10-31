## Image-to-Image Generator with Stable Diffusion and Streamlit

### Overview
A simple Streamlit web app that transforms an input image into a new image guided by a text prompt (Image-to-Image) using Stable Diffusion v1.5 from Hugging Face Diffusers.

### Features
- **Image-to-Image**: Upload an image and enter a prompt to get an enhanced/modified result.
- **Simple UI**: Streamlit-based interface for quick local runs.
- **GPU Acceleration**: Automatically uses CUDA if available; falls back to CPU otherwise.

### Project Structure
- `app.py`: Main Streamlit app that loads the `StableDiffusionImg2ImgPipeline` and renders the UI.

### Requirements
- Python 3.9+ recommended.
- Python dependencies:
  - `diffusers`
  - `torch` (CUDA optional)
  - `Pillow`
  - `streamlit`
  - You may also need: `transformers`, `accelerate`, `safetensors`

Note: Installing `torch` varies by OS and GPU support. Refer to official PyTorch instructions if needed.

### Installation
1) Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install PyTorch for your system:
- CPU only (Windows/Python):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
- CUDA 12.x (if you have a compatible NVIDIA GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

3) Install remaining dependencies:
```bash
pip install diffusers streamlit pillow transformers accelerate safetensors
```

### Run
From the project root, start Streamlit:
```bash
streamlit run app.py
```
You will get a local URL in the terminal (typically `http://localhost:8501`). Open it in your browser.

### How to Use
1) In the app UI:
   - Upload an input image (PNG/JPG/JPEG). It will be resized to 512×512 internally.
   - Enter a prompt describing the desired modification/style.
   - Click "generate image".
2) The result is also saved as `output.png` in the project root, in addition to being displayed in the page.

### Tweak Parameters
You can adjust values in `app.py` as needed:
- `strength`: How strongly the model alters the original image (default 0.65).
- `guidance_scale`: How strictly the model follows the prompt (default 7.5).
- `model_id`: Currently `runwayml/stable-diffusion-v1-5`; can be swapped for other supported models.

### Better Performance with GPU
The app selects device automatically:
```python
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
```
With an NVIDIA GPU and proper CUDA setup, you’ll get significant speedups. Ensure your `torch` install matches your CUDA version.

### Troubleshooting
- "CUDA out of memory":
  - Close other GPU-heavy apps.
  - Try CPU mode or reduce `strength`/resolution.
- Very slow on CPU:
  - Prefer running on a GPU; CPU can be much slower.
- Model download issues:
  - Check your internet connection and any firewall/proxy blocking Hugging Face.

### Model and Notices
- Model: `runwayml/stable-diffusion-v1-5` via Hugging Face Diffusers.
- Review licensing/usage terms for the model and libraries before commercial use.




