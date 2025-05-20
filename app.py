import gradio as gr
from fastai.vision.all import *

learn = load_learner('model.pkl')

def predict_image(img):
    try:
        pred, pred_idx, probs = learn.predict(PILImage.create(img))
        labels = learn.dls.vocab
        return {labels[i]: float(probs[i]) for i in range(len(labels))}
    except Exception as e:
        print(f"HATA: {e}")
        return "Resim analiz edilemedi."

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Sneaker Image"),
    outputs=gr.Label(num_top_classes=3, label="Top Predictions"),
    title="Sneaker Classification AI",
    description="Upload a sneaker image and get predictions from a model trained on 50 sneaker categories."
)

demo.launch()