import gradio as gr
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
material_list = []

with open("results.txt", "r") as f:
    results = f.readlines()
    for line in results: 
        material = line.split(" [")[0]
        material_list.append(material.strip())  # Trim any leading/trailing whitespace
    f.close()
    
text = tokenizer(material_list)

def process_image(image_input):

    results = {}
    float_values = []
    image = preprocess(image_input).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

    counter = 0
    for row in text_probs:
        for column in row:
            float_values.append(float(column))
            results[float(column)] = material_list[counter] 
            counter += 1

    sorted_float_values = sorted(float_values, reverse=True)
    print(sorted_float_values)
    return [["Material : " + str(results[sorted_float_values[0]]), "Confidence : " + str(sorted_float_values[0])], ["Material : " + str(results[sorted_float_values[1]]), "Confidence : " + str(sorted_float_values[1])]]

inputs = gr.inputs.Image(type="pil")
outputs = [gr.outputs.Textbox(label="Top Result"), gr.outputs.Textbox(label="Second Result")]

interface = gr.Interface(fn=process_image, inputs=inputs, outputs=outputs)
interface.launch(share=True)
