# ===============================================================
# A Generative Approach to Fruit Freshness Classification
# ===============================================================

import os
import re
import json
from typing import Tuple, Optional, List

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms



MODEL_PATH = "/content/mobilenet_final.pth"
CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']

IMG_SIZE = 224
IMG2IMG_SIZE = 512


MAX_TIMELINE_STEPS = 12  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



st.set_page_config(layout="wide")

st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #1e1e2f, #2a2a40);
}

/* Card style */
.block-container {
    padding-top: 2rem;
}

/* Chat bubbles */
.user-msg {
    background: #4CAF50;
    color: white;
    padding: 12px 18px;
    border-radius: 20px 20px 0px 20px;
    margin: 8px 0;
    max-width: 85%;
    animation: fadeIn 0.3s ease-in-out;
}
.bot-msg {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(6px);
    color: white;
    padding: 12px 18px;
    border-radius: 20px 20px 20px 0px;
    margin: 8px 0;
    max-width: 85%;
    animation: fadeIn 0.3s ease-in-out;
}

/* Chat role text */
.meta {
    color: #bbb;
    font-size: 11px;
    margin-bottom: 3px;
}

/* Headers */
h1, h2, h3 {
    color: white !important;
    font-weight: 700;
    text-shadow: 0 0 10px rgba(255,255,255,0.2);
}

/* Gradient title */
.st-emotion-cache-1qg05tj h1 {
    background: linear-gradient(to right, #FFD700, #FF8C00);
    -webkit-background-clip: text;
    color: transparent;
}

/* Upload + chat cards */
.upload-card:empty, .chat-card:empty {
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}


/* Fade animation */
@keyframes fadeIn {
    from {opacity:0; transform:translateY(6px);}
    to {opacity:1; transform:translateY(0);}
}
.upload-card:empty,
.chat-card:empty {
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Chat input styling */
.stChatInput {
    margin-top: 15px !important;
}
.st-emotion-cache-75a8hf {
    background: rgba(255,255,255,0.15) !important;
    border-radius: 16px !important;
    padding: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ===============================================================
# IMAGE PREPROCESSING
# ===============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


# ===============================================================
# CLASSIFIER LOADER 
# ===============================================================
@st.cache_resource
def load_classifier():
    try:
        from torchvision import models
        import torch.nn as nn

        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

        if os.path.isfile(MODEL_PATH):
            try:
                # try loading state_dict
                state = torch.load(MODEL_PATH, map_location="cpu")
                model.load_state_dict(state)
            except:
                # fallback: load full model
                model = torch.load(MODEL_PATH, map_location="cpu")

        model = model.float().to(device)
        model.eval()
        return model

    except Exception as e:
        st.warning(f"Classifier failed: {e}")
        return None



classifier = load_classifier()


# ===============================================================
# CONTROLNET + SD1.5 LOADER 
# ===============================================================
SD_PIPE = None
CANNY_DETECTOR = None


@st.cache_resource
def load_sd15_controlnet():
    try:
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        token = os.environ.get("HF_TOKEN")
        token_kw = {"use_auth_token": token} if token else {}

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=dtype,
            **token_kw
        )

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=dtype,
            **token_kw
        )

        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))

        pipe = pipe.to(device)

        try:
            from controlnet_aux import CannyDetector
            detector = CannyDetector()
        except:
            detector = None

        return pipe, detector

    except Exception as e:
        st.error(f"Failed to load SD1.5 + ControlNet: {e}")
        return None, None


def ensure_sd_pipeline():
    global SD_PIPE, CANNY_DETECTOR
    if SD_PIPE is None:
        SD_PIPE, CANNY_DETECTOR = load_sd15_controlnet()
    return SD_PIPE is not None


# ===============================================================
# FALLBACK CANNY
# ===============================================================
def canny_opencv(img: Image.Image) -> Image.Image:
    import numpy as np, cv2
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)


import io

def stylize_image(input_img: Image.Image, style: str) -> Image.Image:
    """
    Run img2img using your SD_PIPE. Returns stylized PIL.Image.
    """
   
    if not ensure_sd_pipeline():
       
        return input_img

    
    img = input_img.resize((IMG2IMG_SIZE, IMG2IMG_SIZE))

 
    style_prompts = {
        "anime": (
            "High-quality anime style food photograph of a banana, "
            "vibrant clean shading, cinematic lighting, crisp details, "
            "soft rim light, 85mm lens, photorealistic anime render. "
            "KEEP EXACT SHAPE AND BACKGROUND. Only change color/style of skin."
        ),
        "pixel": (
            "Pixel art of a banana, high-resolution pixel painting, "
            "clean silhouette, limited palette, 32-bit style, crisp pixels, "
            "transparent-ish background preserved. KEEP SHAPE & BACKGROUND."
        ),
        "neon": (
            "Neon cyberpunk stylized food photo, banana with neon reflections, "
            "dramatic colored rim lights, high contrast, grainy film texture, "
            "sci-fi vibe. KEEP EXACT SHAPE AND BACKGROUND. Only modify colors/textures."
        ),
    }

    prompt = style_prompts.get(style, style_prompts["anime"])

    # create control image (Canny)
    control_img = CANNY_DETECTOR(img) if CANNY_DETECTOR else canny_opencv(img)

    
    out = SD_PIPE(
        prompt=prompt,
        image=img,
        control_image=control_img,
        strength=0.6,                    
        guidance_scale=9.0,            
        controlnet_conditioning_scale=0.7,
        num_inference_steps=30,
        
    ).images[0]

    return out
# ===============================================================
# SESSION STATE INIT 
# ===============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "image" not in st.session_state:
    st.session_state.image = None
if "stage" not in st.session_state:
    st.session_state.stage = None

# --- UI: Stylize controls ---
if st.session_state.image:
    st.markdown("### 🎨 Stylize (fun modes)")
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        style_choice = st.selectbox("Choose style", ["anime", "pixel", "neon"])
    with col_b:
        strength_slider = st.slider("Strength (img2img)", 0.0, 1.0, 0.6, 0.05)
    with col_c:
        go = st.button("Generate Stylized Image")

    if go:
        orig_img = st.session_state.image
        
        with st.spinner("Generating stylized image..."):
          
            def stylize_with_strength(img, style, strength_val):
                if not ensure_sd_pipeline():
                    return img
                img_resized = img.resize((IMG2IMG_SIZE, IMG2IMG_SIZE))
                control_img = CANNY_DETECTOR(img_resized) if CANNY_DETECTOR else canny_opencv(img_resized)
                prompt = {
                    "anime": (
                        "High-quality anime style food photograph of a banana, "
                        "vibrant clean shading, cinematic lighting, crisp details, "
                        "soft rim light, 85mm lens, photorealistic anime render. "
                        "KEEP EXACT SHAPE AND BACKGROUND. Only change color/style of skin."
                    ),
                    "pixel": (
                        "Pixel art of a banana, high-resolution pixel painting, "
                        "clean silhouette, limited palette, 32-bit style, crisp pixels, "
                        "transparent-ish background preserved. KEEP SHAPE & BACKGROUND."
                    ),
                    "neon": (
                        "Neon cyberpunk stylized food photo, banana with neon reflections, "
                        "dramatic colored rim lights, high contrast, grainy film texture, "
                        "sci-fi vibe. KEEP EXACT SHAPE AND BACKGROUND. Only modify colors/textures."
                    ),
                }.get(style, "")

                out = SD_PIPE(
                    prompt=prompt,
                    image=img_resized,
                    control_image=control_img,
                    strength=float(strength_val),
                    guidance_scale=9.0,
                    controlnet_conditioning_scale=0.7,
                    num_inference_steps=30,
                ).images[0]
                return out

            stylized = stylize_with_strength(orig_img, style_choice, strength_slider)

       
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Original")
            st.image(orig_img, use_column_width=True)
        with c2:
            st.caption(f"Stylized — {style_choice}")
            st.image(stylized, use_column_width=True)

           
            buf = io.BytesIO()
            stylized.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                label="Download Stylized PNG",
                data=buf,
                file_name=f"banana_stylized_{style_choice}.png",
                mime="image/png"
            )

# ===============================================================
# IMAGE GENERATOR 
# ===============================================================
def predict_future(input_img: Image.Image, current_stage: str, days: int):
    # -----------------------------------------
    # 1. STAGE → BASE RIPENESS %
    # -----------------------------------------
    def stage_to_ripeness(stage):
        mapping = {
            "unripe": 0,
            "early-ripe": 30,
            "ripe": 50,
            "overripe": 75,
            "rotten": 100,
        }
        return mapping.get(stage, 0)

    base_ripeness = stage_to_ripeness(current_stage)

    # Each 2 days ≈ +10% ripeness progression
    future_ripeness = min(100, base_ripeness + (days * 5))
    ripeness = future_ripeness

    # -----------------------------------------
    # 2. DETERMINE TARGET STAGE + DESCRIPTION
    # 
    # -----------------------------------------
    if ripeness < 20:
        desc = (
            "very unripe banana with hard firm green skin, no yellowing, "
            "smooth matte texture"
        )
        stage_name = "unripe"

    elif ripeness < 40:
        desc = (
            "early-ripening banana with green-yellow mixed skin, slight warm tint, "
            "tiny freckles beginning to form"
        )
        stage_name = "early-ripe"

    elif ripeness < 60:
        desc = (
            "ripe banana with bright clean yellow skin, smooth glossy texture, "
            "no brown spots"
        )
        stage_name = "ripe"

    elif ripeness < 80:
        desc = (
            "overripe banana with many dark brown freckles, soft texture, "
            "darker yellow skin"
        )
        stage_name = "overripe"

    else:
        desc = (
            "rotten banana with blackened skin, collapsed mushy texture, mold patches, "
            "heavy dark bruising and decay"
        )
        stage_name = "rotten"

    # -----------------------------------------
    # 3. STRONG DIFFUSION PROMPT
    # -----------------------------------------
    prompt = (
        f"Ultra-realistic banana at {ripeness}% ripeness. "
        f"Appearance: {desc}. "
        "8K macro food photography, extremely detailed texture. "
        "KEEP EXACT SHAPE AND BACKGROUND. "
        "Only modify banana skin color, ripeness indicators, and texture. "
        "If rotten: exaggerate mold, bruising, blackening, and mushiness."
    )

    # -----------------------------------------
    # 4. CONTROLNET IMAGE-TO-IMAGE
    # -----------------------------------------
    if not ensure_sd_pipeline():
        return input_img, stage_name   # fallback

    img = input_img.resize((IMG2IMG_SIZE, IMG2IMG_SIZE))
    control_img = CANNY_DETECTOR(img) if CANNY_DETECTOR else canny_opencv(img)

    out = SD_PIPE(
        prompt=prompt,
        image=img,
        control_image=control_img,
        strength=0.55,
        guidance_scale=8.5,
        controlnet_conditioning_scale=0.55,
        num_inference_steps=25
    ).images[0]

    # -----------------------------------------
    # 5. RETURN IMAGE + STAGE NAME ONLY
    # -----------------------------------------
    return out, stage_name



# ===============================================================
# PHI-3 LLM PARSER 
# ===============================================================
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
        LLM = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        return True

    except Exception as e:
        st.error(f"LLM failed: {e}")
        return False


def parse_intent_llm(msg: str) -> dict:
    if not ensure_llm():
        return {"intent": "unknown", "days": 2}

    prompt = (
        "Return only JSON:\n"
        "{intent: future_image|nutrition|edibility|status|timeline|unknown, days: int}\n"
        f"User: {msg}\nJSON:"
    )

    inp = TOKENIZER(prompt, return_tensors="pt").to(LLM.device)
    out = LLM.generate(**inp, max_new_tokens=80, temperature=0.2)
    txt = TOKENIZER.decode(out[0], skip_special_tokens=True)

    try:
        j = re.search(r"\{.*\}", txt, re.S).group()
        return json.loads(j)
    except:
        return {"intent": "unknown", "days": 2}


# ===============================================================
# RULE-BASED INTENT PARSER 
# ===============================================================

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
    if "stage" in msg_l:
        return {"intent": "status", "days": days}
    if "nutrition" in msg_l or "advice" in msg_l:
        return {"intent": "nutrition", "days": days}
    if "older" in msg_l or "after" in msg_l or "days" in msg_l:
        return {"intent": "future_image", "days": days}
    return {"intent": "unknown", "days": days}


# ===============================================================
# STATIC FRUIT INFO
# ===============================================================

def check_edibility(stage: str) -> str:
    details = {
        "ripe": (
            "🍌 **Yes, it's perfectly edible!**\n"
            "\n"
            "Soft, sweet, and ready to eat."
        ),

        "overripe": (
            "🍌 **Yes, still edible!**\n"
            "\n"
            "Very sweet and mushy — best for smoothies, shakes, or baking."
        ),

        "unripe": (
            "🍌 **Edible but not recommended right now.**\n"
            "\n"
            "Hard, starchy, and not sweet yet. Let it ripen for a better taste."
        ),

        "rotten": (
            "⚠️ **Not edible — unsafe to eat!**\n"
            "\n"
            "Rotten bananas may contain mold, bacteria, or fermentation gases.\n"
            "❌ Please throw it away."
        ),
    }

    return details.get(stage, "I couldn't determine the edibility of this fruit.")


   



def nutrition(stage: str) -> str:
    info = {
        "ripe": (
            "🥗 **Nutrition — Ripe Banana**\n\n"
            "- Rich in **potassium** (supports heart health)\n"
            "- Good source of **Vitamin B6** (brain & metabolism)\n"
            "- Natural **quick-release sugars** for instant energy\n"
            "- Contains **dietary fiber** for smooth digestion\n\n"
            "👉 Perfect for breakfast, workouts, or a quick snack."
        ),

        "unripe": (
            "🥗 **Nutrition — Unripe Banana**\n\n"
            "- High in **resistant starch** (great for gut microbiome)\n"
            "- Lower sugar content than ripe bananas\n"
            "- Helps regulate **blood sugar levels**\n\n"
            "👉 Best used for boiling, cooking, or digestive benefits."
        ),

        "overripe": (
            "🥗 **Nutrition — Overripe Banana**\n\n"
            "- High in **antioxidants** released during ripening\n"
            "- Very soft and **easy to digest**\n"
            "- Sweeter because starch converts into sugar\n\n"
            "👉 Ideal for baking, smoothies, pancakes, and natural sweeteners."
        ),

        "rotten": (
            "⚠️ **Nutrition — Rotten Banana**\n\n"
            "- May contain **harmful bacteria**, mold, or fermentation\n"
            "- **Unsafe to consume**\n\n"
            "❌ Please discard immediately."
        )
    }

    return info.get(stage, "No nutrition info available for this stage.")





# ===============================================================
# UTIL: build timeline steps (0..N step 2, include final day if odd)
# ===============================================================

def build_timeline_steps(days: int) -> List[int]:
    if days <= 0:
        return [0]

    # limit days to a reasonable upper bound
    days = min(days, 40)

    if days <= 2:
        steps = [0, days] if days != 0 else [0]
    else:
        steps = list(range(0, days + 1, 2))
        if days not in steps:
            steps.append(days)

    # ensure steps unique & sorted
    steps = sorted(list(dict.fromkeys(steps)))

    # cap number of images
    if len(steps) > MAX_TIMELINE_STEPS:
        
        keep = [steps[0]]
        middle_count = MAX_TIMELINE_STEPS - 2
        if middle_count > 0:
            idxs = [int(i) for i in
                    list(torch.linspace(1, len(steps) - 2, steps=middle_count).tolist())]
            for i in idxs:
                keep.append(steps[i])
        keep.append(steps[-1])
        steps = sorted(list(dict.fromkeys(keep)))

    return steps


# ===============================================================
# MAIN UI — MODERNIZED 
# ===============================================================

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "image" not in st.session_state:
    st.session_state.image = None
if "stage" not in st.session_state:
    st.session_state.stage = None

st.title("A Generative Approach to Fruit Freshness Classification")

col1, col2 = st.columns([1, 1], gap="large")

# ---------------------------
# LEFT SIDE — Upload Card
# ---------------------------
with col1:
    st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
    st.header("📤 Upload Fruit Image")

    file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded", use_column_width=True)
        st.session_state.image = img

        if classifier:
            x = transform(img).unsqueeze(0).to(device).float()
            clf = classifier.to(device).float()
            with torch.no_grad():
                out = clf(x)

            p = torch.softmax(out, dim=1).cpu().numpy().squeeze()
            idx = int(p.argmax())
            stage = CLASS_NAMES[idx]

            st.session_state.stage = stage
            st.success(f"Stage: **{stage}** ({p[idx] * 100:.1f}%") 

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# RIGHT SIDE — Chat Card
# ---------------------------
with col2:
    st.markdown("<div class='chat-card'>", unsafe_allow_html=True)
    st.header("💬 Know Your Fruit")

    use_llm = st.checkbox("Use Phi-3 Intent Parser", value=False)

    box = st.empty()

    def render():
        with box.container():
            for role, text in st.session_state.messages:
                css = "user-msg" if role == "user" else "bot-msg"
                st.markdown(f"<div class='meta'>{role.upper()}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='{css}'>{text}</div>", unsafe_allow_html=True)

    render()

    msg = st.chat_input("Ask something about the fruit...")

    if msg:
        st.session_state.messages.append(("user", msg))
        render()

        if not st.session_state.image:
            st.session_state.messages.append(("assistant", "Please upload an image first 😊"))
            render()
        else:
            parsed = parse_intent_llm(msg) if use_llm else parse_intent_rule(msg)
            intent, days = parsed.get("intent"), parsed.get("days", 2)
            stg, img = st.session_state.stage, st.session_state.image

            if intent == "future_image":
                st.session_state.messages.append(("assistant", f"Generating how the banana looks after **{days} days**..."))
                render()

                with st.spinner("🧪 Transforming banana..."):
                    new_img, new_stage = predict_future(img, stg, days)

                st.session_state.messages.append(("assistant", f"Predicted stage: **{new_stage}**"))
                render()

                # --- Side-by-side comparison ---
                c1, c2 = st.columns(2)

                with c1:
                    st.subheader("📍 Current Image")
                    st.image(img, caption="Original", use_column_width=True)

                with c2:
                    st.subheader(f"⏳ After {days} Days")
                    st.image(new_img, caption=f"Stage: {new_stage}", use_column_width=True)

                import io
                buf = io.BytesIO()
                new_img.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="📥 Download Image",
                    data=byte_im,
                    file_name=f"banana_{days}_days.png",
                    mime="image/png"
                )


            elif intent == "status":
                st.session_state.messages.append(("assistant", f"The fruit looks **{stg}**."))
                render()

            elif intent == "edibility":
                st.session_state.messages.append(("assistant", check_edibility(stg)))
                render()

            elif intent == "nutrition":
                st.session_state.messages.append(("assistant", nutrition(stg)))
                render()

            elif intent == "timeline":
                steps = build_timeline_steps(days)

                st.session_state.messages.append(
                    ("assistant", f"Generating timeline for next **{days} days**...")
                )
                render()

                # Generate images
                imgs_and_stages = []
                with st.spinner("🧪 Generating timeline images..."):
                    for d in steps:
                        t_img, t_stage = predict_future(img, stg, d)
                        imgs_and_stages.append((d, t_img, t_stage))

                # 🔥 Sort images by day (IMPORTANT)
                imgs_and_stages = sorted(imgs_and_stages, key=lambda x: x[0])

                with st.expander(f"Ripening timeline — 0 → {steps[-1]} days", expanded=True):
                    per_row = 6
                    rows = [imgs_and_stages[i:i+per_row] for i in range(0, len(imgs_and_stages), per_row)]
                    for row in rows:
                        cols = st.columns(len(row))
                        for col, (d, t_img, t_stage) in zip(cols, row):
                            with col:
                                st.image(t_img, caption=f"Day {d} → {t_stage}")

                


               
                summary = "\n".join([f"Day {d} → {s}" for d,_,s in imgs_and_stages])
                st.session_state.messages.append(("assistant", f"📅 **Timeline Summary**\n{summary}"))
                render()

            else:
                st.session_state.messages.append(("assistant", "Not sure 🤔 Try: '3 days older', 'Is it edible?', 'nutrition advice', or 'timeline for 8'."))
                render()

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("NSFW safety disabled. Classifier FP32 safe. SD15 + ControlNet FP16.")