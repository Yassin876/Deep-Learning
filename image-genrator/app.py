from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import streamlit as st
#load model
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {pipe.device} device")
print(f"model loaded successfully")

#generate image function
def generate_image(init_image,prompt):
    init_image = Image.open(init_image).convert("RGB")
    init_image = init_image.resize((512, 512))
    image = pipe(prompt=prompt, image=init_image, strength=0.65, guidance_scale=7.5).images[0]
    image.save("output.png")
    return image

#streamlit app ui
st.title("Stable Diffusion Image to Image")
init_image = st.file_uploader("upload image",type=["png","jpg","jpeg"])
prompt = st.text_input("enter prompt")
button = st.button("generate image")
if button:
    if init_image is not None and prompt:
        image = generate_image(init_image, prompt)
        st.image(image, caption="generated image", use_column_width=True)
    else:
        st.error("please upload an image and enter a prompt")