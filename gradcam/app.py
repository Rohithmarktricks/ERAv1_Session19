import gradio as gr
import clip
import torch

import utils

#clip_model = "RN50x4"
clip_model = "RN50x64"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_model, device=device, jit=False)
model.eval()


def grad_cam_fn(text, img, saliency_layer):
    resize = model.visual.input_resolution
    img = img.resize((resize, resize))

    text_input = clip.tokenize([text]).to(device)
    text_feature = model.encode_text(text_input).float()
    image_input = preprocess(img).unsqueeze(0).to(device)

    attn_map = utils.gradCAM(
        model.visual,
        image_input,
        text_feature,
        getattr(model.visual, saliency_layer)
    )
    attn_map = attn_map.squeeze().detach().cpu().numpy()
    attn_map = utils.getAttMap(img, attn_map)

    return attn_map


interface = gr.Interface(
    fn=grad_cam_fn,
    inputs=[
        gr.inputs.Textbox(
            label="Target Text",
            lines=1),
        gr.inputs.Image(
            label='Input Image',
            image_mode="RGB",
            type='pil',
            shape=(512, 512)),
        gr.inputs.Dropdown(
            ["layer4", "layer3", "layer2", "layer1"],
            default="layer4",
            label="Saliency Layer")
    ],
    outputs=gr.outputs.Image(
        type="pil", 
        label="Attention Map"),
    examples=[
        ['a cat lying on the floor', 'assets/cat_dog.jpg', 'layer4'],
        ['a dog sitting', 'assets/cat_dog.jpg', 'layer4']
    ],
    description="OpenAI CLIP Grad CAM")
interface.launch()
