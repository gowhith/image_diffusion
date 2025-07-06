
# 🎨 Custom Image Generation with Diffusion Models

This project demonstrates **Text-to-Image Generation** using **Stable Diffusion** powered by Hugging Face's Diffusers library. Users provide a text prompt, and the model generates a high-quality image aligned with the description.

---

## ✅ Project Overview

| Component           | Technology/Library                     | Purpose                                      |
|---------------------|----------------------------------------|----------------------------------------------|
| Programming Language | Python                                | Core development                             |
| Deep Learning       | PyTorch + Hugging Face Diffusers       | Diffusion model loading & inference          |
| Generative Model    | Stable Diffusion v1.4 (Hugging Face)  | State-of-the-art text-to-image generation    |
| Dataset (Optional)  | LAION, MS-COCO, Open Images           | For fine-tuning custom models (advanced)     |
| Image Processing    | Pillow (PIL), OpenCV                  | Post-processing generated images             |
| UI (Optional)       | Streamlit or Gradio                   | Interactive frontend for user input          |
| Hardware Requirement| GPU (Google Colab recommended)        | Efficient image generation                   |

---

## ✅ Step-by-Step Project Workflow

### **1. Setup Requirements**
- Hugging Face Account: [https://huggingface.co](https://huggingface.co)
- Accept license for Stable Diffusion model: [https://huggingface.co/CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- Generate Hugging Face Access Token: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

### **2. Where to Run Each Part**

| Task                        | Recommended Platform | Reason                            |
|-----------------------------|----------------------|------------------------------------|
| Model Loading & Generation  | Google Colab         | Free GPU, fast image processing   |
| Local UI Testing (Optional) | Laptop (Low spec OK) | Streamlit/Gradio UI development   |
| Fine-Tuning (Advanced)      | Google Colab Pro     | High compute resources required   |

---

### **3. Image Generation via Google Colab**

Example Colab code:

```python
!pip install diffusers transformers accelerate scipy safetensors

from huggingface_hub import login
login('YOUR_HF_ACCESS_TOKEN')

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16
).to("cuda")

prompt = "Futuristic cityscape at sunset"
image = pipe(prompt, guidance_scale=7.5).images[0]

image.show()
image.save("output.png")
```

Generated images can be downloaded from Colab's file sidebar.

---

### **4. Optional: UI with Streamlit**

Basic Streamlit setup (runs on laptop):

```bash
pip install streamlit diffusers torch transformers pillow
```

Example app:

```python
import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

st.title("AI Image Generator")
prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to("cpu")  # CPU-only mode for low-resource devices
    image = pipe(prompt).images[0]
    st.image(image, caption="Generated Image")
```

Run with:

```bash
streamlit run app.py
```

⚠ Note: CPU generation is extremely slow; Colab is recommended for actual image generation.

---

## ✅ Explanation of Notebook Issue

Your original notebook (`main.ipynb`) failed to render on GitHub due to:

```
Invalid Notebook: The 'state' key is missing from 'metadata.widgets'.
```

This happens when:

- Notebooks are saved improperly (abrupt shutdowns, kernel crashes).
- Metadata for Jupyter widgets becomes incomplete or corrupted.
- Version mismatches between `nbformat` or `nbconvert`.

**Impact:**
- Notebook previews break on platforms like GitHub.
- Core code may still work if manually corrected or run in Colab/Jupyter.

---

## ✅ How to Fix Notebook Corruption

- Open notebook in a text editor, remove `"metadata.widgets"` sections.
- Use `nbformat` Python package to clean notebook:
  
```python
import nbformat
with open("main.ipynb") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if 'widgets' in cell.get('metadata', {}):
        del cell['metadata']['widgets']

with open("fixed_main.ipynb", "w") as f:
    nbformat.write(nb, f)
```

- Alternatively, re-upload notebook to Google Colab, resave it.

---

## ✅ Future Best Practices

✔ Always save notebooks properly before closing  
✔ Avoid abrupt kernel shutdowns  
✔ Test notebooks locally and in Colab regularly  
✔ Keep `nbformat` and `nbconvert` updated  

---

## ✅ Project Notes

- Model fine-tuning requires advanced GPU resources; use Colab Pro.
- Image quality improves with prompt engineering and parameter tuning (`guidance_scale`, `inference_steps`).
- Generated images are for research/demo purposes; commercial use may require additional licenses.

---

# 🎉 **End of README**
