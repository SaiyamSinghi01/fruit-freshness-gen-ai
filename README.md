# A Generative Approach to Fruit Freshness Classification 🍌✨

This project implements a **hybrid Generative + Discriminative AI system** capable of:

- Ripeness **classification**
- **Future appearance prediction** using diffusion models
- **Ripening timeline simulation**
- **Creative stylization** (Anime / Pixel / Neon)
- **Natural-language interaction** through LLMs

For controlled biological modeling, the system focuses on **one fruit category** with predictable ripening behaviour. After establishing this rationale, bananas are used because they show clear visual transitions across four stages:

- **Unripe**
- **Ripe**
- **Overripe**
- **Rotten**

---

## 🚀 Features

### 1️⃣ Ripeness Classification
MobileNet-V2 classifier predicts one of four stages:
- Unripe  
- Ripe  
- Overripe  
- Rotten  

### 2️⃣ Future Appearance Prediction
Uses **Stable Diffusion 1.5 + ControlNet (Canny)** to simulate how the fruit will look after *N* days.


### 4️⃣ Creative Stylization
Transforms the uploaded image into:
- 🎎 Anime Style  
- 🟦 Pixel Art  
- 🌃 Neon Cyberpunk  

### 5️⃣ Natural-Language Interaction
You can ask:
- *How will it look after 5 days?*
- *Is it edible?*
- *Generate 8-day timeline.*

Uses **Phi-3 Mini**, with a **fallback regex parser**.

### 6️⃣ Downloadable Output
All generated images can be downloaded as PNG files.

---





### 3️⃣ Ripening Timeline Simulation
Generates multiple steps:
