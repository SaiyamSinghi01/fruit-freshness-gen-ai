# app.py
# A Generative Approach to Fruit Freshness Classification — Streamlit app
# Features included:
# - MobileNet-V2 classifier loader (FP32)
# - Rule-based + Phi-3 LLM intent parser (optional)
# - Stable Diffusion 1.5 + ControlNet img2img pipeline hooks (optional)
# - Predict future appearance, timeline generator, stylizer (anime/pixel/neon)
# - Side-by-side comparison and download buttons

import os
import re
import json
import io
from typing import Tuple, List
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms

# -------------------------
# Config
# -------------------------
MODEL_PATH = "mobilenet_final.pth"   # put your classifier here
CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']
IMG_SIZE = 224
IMG2IMG_SIZE = 512
MAX_TIMELINE_STEPS = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# UI config
# -------------------------
st.set_page_config(layout="wide", page_title="Generative Fruit Freshness")
st.title("A Generative Approach to Fruit Freshness Classification")

# minimal CSS to keep nice visuals (optional)
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# -------------------------
# Classifier loader
# -------------------------
@st.cache_resource
def load_classifier():
    try:
        from torchvision import models
        import torch.nn as nn
        model = models.mobilenet_v2(pretrained=False)
        # adapt classifier
        model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
        if os.path.isfile(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model = model.float().to(device)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Could not load classifier: {e}")
        return None

classifier = load_classifier()

# -------------------------
# Stable Diffusion + ControlNet (lazy load)
# -------------------------
SD_PIPE = None
CANNY_DETECTOR = None

@st.cache_resource
def load_sd15_controlnet():
    try:
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        token = os.environ.get("HF_TOKEN")
        token_kw = {"use_auth_token": token} if token else {}
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype, **token_kw)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype, **token_kw)
        # disable safety checker (optional/hacky)
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
        pipe = pipe.to(device)
        try:
            from controlnet_aux import CannyDetector
            detector = CannyDetector()
        except Exception:
            detector = None
        return pipe, detector
    except Exception as e:
        st.warning("SD/ControlNet not available or failed to load. Future/stylize features will be disabled.")
        return None, None

def ensure_sd_pipeline():
    global SD_PIPE, CANNY_DETECTOR
    if SD_PIPE is None:
        SD_PIPE, CANNY_DETECTOR = load_sd15_controlnet()
    return SD_PIPE is not None

def canny_opencv(img: Image.Image) -> Image.Image:
    try:
        import numpy as np
        import cv2
        gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges)
    except Exception:
        return img

# -------------------------
# LLM (Phi-3) loader (optional)
# -------------------------
LLM = None
TOKENIZER = None

def ensure_llm():
    global LLM, TOKENIZER
    if LLM:
        return True
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        name = "microsoft/phi-3-mini-4k-instruct"
        TOKENIZER = AutoTokenizer.from_pretrained(name)
        LLM = AutoModelForCausalLM.from_pretrained(name,
                                                   torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                                   device_map="auto")
        return True
    except Exception:
        return False

def parse_intent_llm(msg: str) -> dict:
    if not ensure_llm():
        return {"intent": "unknown", "days": 2}
    prompt = ("Return only JSON:\n"
              "{intent: future_image|nutrition|edibility|status|timeline|unknown, days: int}\n"
              f"User: {msg}\nJSON:")
    inp = TOKENIZER(prompt, return_tensors="pt").to(LLM.device)
    out = LLM.generate(**inp, max_new_tokens=80, temperature=0.2)
    txt = TOKENIZER.decode(out[0], skip_special_tokens=True)
    try:
        j = re.search(r"\{.*\}", txt, re.S).group()
        return json.loads(j)
    except:
        return {"intent": "unknown", "days": 2}

# -------------------------
# Rule-based fallback parser
# -------------------------
def parse_intent_rule(msg: str) -> dict:
    msg_l = msg.lower()
    days = 2
    m = re.search(r"(\d+)", msg_l)
    if m:
        try:
            days = int(m.group(1))
        except:
            days = 2
    if "timeline" in msg_l or "timel" in msg_l:
        return {"intent": "timeline", "days": days}
    if "edible" in msg_l or "eat" in msg_l:
        return {"intent": "edibility", "days": days}
    if "stage" in msg_l or "what is" in msg_l:
        return {"intent": "status", "days": days}
    if "nutrition" in msg_l or "advice" in msg_l:
        return {"intent": "nutrition", "days": days}
    if "older" in msg_l or "after" in msg_l or "days" in msg_l:
        return {"intent": "future_image", "days": days}
    return {"intent": "unknown", "days": days}

# -------------------------
# Info modules
# -------------------------
def check_edibility(stage: str) -> str:
    details = {
        "ripe": "🍌 Yes — it's edible. Soft, sweet, ready to eat.",
        "overripe": "🍌 Yes — edible. Very soft and sweet; great for baking/smoothies.",
        "unripe": "🍌 Edible but firm and starchy; not recommended to eat raw.",
        "rotten": "⚠️ Not edible — unsafe. Discard immediately."
    }
    return details.get(stage, "I couldn't determine the edibility of this fruit.")

def nutrition(stage: str) -> str:
    info = {
        "ripe": ("Nutrition — Ripe: Rich in potassium, Vitamin B6, natural sugars, fiber."),
        "unripe": ("Nutrition — Unripe: Higher resistant starch, lower sugar, gut-friendly."),
        "overripe": ("Nutrition — Overripe: Higher sugars, easier digestion, good for baking."),
        "rotten": ("Rotten: Unsafe — may contain harmful microbes; no nutritional benefit.")
    }
    return info.get(stage, "No nutrition info available for this stage.")

# -------------------------
# Predict future (image generator)
# Returns (PIL.Image, stage_name)
# -------------------------
def predict_future(input_img: Image.Image, current_stage: str, days: int) -> Tuple[Image.Image, str]:
    # Map stage to base ripeness percent
    def stage_to_ripeness(stage):
        mapping = {
            "unripe": 0,
            "early-ripe": 30,
            "ripe": 50,
            "overripe": 75,
            "rotten": 100,
        }
        return mapping.get(stage, 0)
    base = stage_to_ripeness(current_stage)
    # progression rule: +5% per day (adjustable)
    future = min(100, base + days * 5)
    ripeness = int(future)

    # choose stage name
    if ripeness < 20:
        desc = "very green, firm, matte skin"
        stage_name = "unripe"
    elif ripeness < 40:
        desc = "green-yellow mixed skin, slight freckles"
        stage_name = "early-ripe"
    elif ripeness < 60:
        desc = "bright yellow skin, smooth"
        stage_name = "ripe"
    elif ripeness < 80:
        desc = "yellow with brown spots, soft"
        stage_name = "overripe"
    else:
        desc = "blackened patches, mushy and moldy"
        stage_name = "rotten"

    # strong prompt for SD
    prompt = (
        f"Ultra-realistic food photo of a banana at {ripeness}% ripeness. Appearance: {desc}. "
        "8K macro food photography, extremely detailed texture. KEEP EXACT SHAPE AND BACKGROUND. "
        "Only modify banana skin color and ripeness indicators. If rotten: show mold and darkening."
    )

    # If SD pipeline absent, return resized original and stage name
    if not ensure_sd_pipeline():
        return input_img.resize((IMG2IMG_SIZE, IMG2IMG_SIZE)), stage_name

    img = input_img.resize((IMG2IMG_SIZE, IMG2IMG_SIZE))
    control_img = CANNY_DETECTOR(img) if CANNY_DETECTOR else canny_opencv(img)

    out = SD_PIPE(
        prompt=prompt,
        image=img,
        control_image=control_img,
        strength=0.55,
        guidance_scale=8.5,
        controlnet_conditioning_scale=0.6,
        num_inference_steps=25
    ).images[0]

    return out, stage_name

# -------------------------
# Stylizer (anime/pixel/neon)
# -------------------------
def stylize_image(input_img: Image.Image, style: str, strength: float = 0.6) -> Image.Image:
    if not ensure_sd_pipeline():
        return input_img
    img = input_img.resize((IMG2IMG_SIZE, IMG2IMG_SIZE))
    style_prompts = {
        "anime": "High-quality anime style food photograph of a banana, cel-shaded, vibrant, cinematic lighting. KEEP EXACT SHAPE AND BACKGROUND.",
        "pixel": "Pixel art of a banana, 32-bit, crisp pixels, limited palette. KEEP SHAPE & BACKGROUND.",
        "neon": "Neon cyberpunk stylized banana, colored rim lights, high contrast, sci-fi vibe. KEEP SHAPE & BACKGROUND."
    }
    prompt = style_prompts.get(style, style_prompts["anime"])
    control_img = CANNY_DETECTOR(img) if CANNY_DETECTOR else canny_opencv(img)
    out = SD_PIPE(
        prompt=prompt,
        image=img,
        control_image=control_img,
        strength=float(strength),
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.7,
        num_inference_steps=28
    ).images[0]
    return out

# -------------------------
# Timeline builder
# -------------------------
def build_timeline_steps(days: int) -> List[int]:
    if days <= 0:
        return [0]
    days = min(days, 40)
    if days <= 2:
        steps = [0, days] if days != 0 else [0]
    else:
        steps = list(range(0, days + 1, 2))
        if days not in steps:
            steps.append(days)
    steps = sorted(list(dict.fromkeys(steps)))
    if len(steps) > MAX_TIMELINE_STEPS:
        keep = [steps[0]]
        middle_count = MAX_TIMELINE_STEPS - 2
        if middle_count > 0:
            idxs = [int(i) for i in list(torch.linspace(1, len(steps)-2, steps=middle_count).tolist())]
            for i in idxs:
                keep.append(steps[i])
        keep.append(steps[-1])
        steps = sorted(list(dict.fromkeys(keep)))
    return steps

# -------------------------
# Session state & UI
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "image" not in st.session_state:
    st.session_state.image = None
if "stage" not in st.session_state:
    st.session_state.stage = "unknown"

col1, col2 = st.columns([1,1], gap="large")

with col1:
    st.header("Upload")
    file = st.file_uploader("Upload image (png/jpg/jpeg)", type=["png","jpg","jpeg"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded", use_column_width=True)
        st.session_state.image = img
        # classify if classifier loaded
        if classifier:
            x = transform(img).unsqueeze(0).to(device).float()
            clf = classifier.to(device).float()
            with torch.no_grad():
                out = clf(x)
            p = torch.softmax(out, dim=1).cpu().numpy().squeeze()
            idx = int(p.argmax())
            stage = CLASS_NAMES[idx]
            st.session_state.stage = stage
            st.success(f"Stage: {stage} ({p[idx]*100:.1f}%)")

with col2:
    st.header("Controls & Chat")
    use_llm = st.checkbox("Use Phi-3 intent parser (optional)", value=False)
    st.write("Ask: 'How will it look after 4 days?', 'Is it edible?', 'Generate timeline for 8' ")

    box = st.empty()
    def render_messages():
        with box.container():
            for role,text in st.session_state.messages:
                if role == "user":
                    st.markdown(f"**You:** {text}")
                else:
                    st.markdown(f"**Bot:** {text}")
    render_messages()

    msg = st.chat_input("Ask about the fruit...")
    if msg:
        st.session_state.messages.append(("user", msg))
        render_messages()
        if not st.session_state.image:
            st.session_state.messages.append(("bot", "Please upload an image first."))
            render_messages()
        else:
            parsed = parse_intent_llm(msg) if use_llm else parse_intent_rule(msg)
            intent, days = parsed.get("intent"), parsed.get("days", 2)
            stg = st.session_state.stage
            img = st.session_state.image

            if intent == "future_image":
                st.session_state.messages.append(("bot", f"Generating how the fruit looks after {days} days..."))
                render_messages()
                with st.spinner("Generating..."):
                    new_img, new_stage = predict_future(img, stg, days)
                st.session_state.messages.append(("bot", new_stage))
                render_messages()
                # side-by-side
                c1,c2 = st.columns(2)
                with c1:
                    st.subheader("Original")
                    st.image(img, use_column_width=True)
                with c2:
                    st.subheader(f"After {days} days — {new_stage}")
                    st.image(new_img, use_column_width=True)
                # download
                buf = io.BytesIO()
                new_img.save(buf, format="PNG")
                buf.seek(0)
                st.download_button("Download Generated", data=buf.getvalue(), file_name=f"generated_{days}d.png", mime="image/png")
            elif intent == "status":
                st.session_state.messages.append(("bot", f"Current stage: {stg}"))
                render_messages()
            elif intent == "edibility":
                st.session_state.messages.append(("bot", check_edibility(stg)))
                render_messages()
            elif intent == "nutrition":
                st.session_state.messages.append(("bot", nutrition(stg)))
                render_messages()
            elif intent == "timeline":
                steps = build_timeline_steps(days)
                st.session_state.messages.append(("bot", f"Generating timeline for {days} days..."))
                render_messages()
                imgs_and_stages = []
                with st.spinner("Generating timeline images..."):
                    for d in steps:
                        t_img, t_stage = predict_future(img, stg, d)
                        imgs_and_stages.append((d, t_img, t_stage))
                imgs_and_stages = sorted(imgs_and_stages, key=lambda x: x[0])
                with st.expander(f"Ripening timeline — 0 → {steps[-1]} days", expanded=True):
                    per_row = 4
                    rows = [imgs_and_stages[i:i+per_row] for i in range(0, len(imgs_and_stages), per_row)]
                    for row in rows:
                        cols = st.columns(len(row))
                        for col, (d, t_img, t_stage) in zip(cols, row):
                            with col:
                                st.image(t_img, caption=f"Day {d} → {t_stage}")
                summary = "\n".join([f"Day {d} → {s}" for d,_,s in imgs_and_stages])
                st.session_state.messages.append(("bot", f"Timeline summary:\n{summary}"))
                render_messages()
            else:
                st.session_state.messages.append(("bot", "I didn't understand. Examples: '3 days older', 'Is it edible?', 'timeline for 8'"))
                render_messages()

# -------------------------
# Stylizer UI (bottom)
# -------------------------
st.markdown("---")
st.header("Stylize (Creative Modes)")
if st.session_state.image:
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        style_choice = st.selectbox("Style", ["anime","pixel","neon"])
    with col_b:
        strength = st.slider("Strength", 0.0, 1.0, 0.6, 0.05)
    with col_c:
        go = st.button("Generate Stylized Image")
    if go:
        with st.spinner("Generating stylized image..."):
            styl = stylize_image(st.session_state.image, style_choice, strength)
        c1,c2 = st.columns(2)
        with c1:
            st.caption("Original")
            st.image(st.session_state.image, use_column_width=True)
        with c2:
            st.caption(f"{style_choice.title()} Stylized")
            st.image(styl, use_column_width=True)
            buf = io.BytesIO()
            styl.save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download Stylized PNG", data=buf.getvalue(), file_name=f"stylized_{style_choice}.png", mime="image/png")
else:
    st.info("Upload an image to use stylizer.")

st.caption("Note: SD/ControlNet and LLM are optional and will be used only if available. Set HF_TOKEN env var if required.")
