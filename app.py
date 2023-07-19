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

def process_image(image_input, material_input):

    results = {}
    float_values = []
    image = preprocess(image_input).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    #print("Label probs:", text_probs) 

    counter = 0
    for row in text_probs:
        for column in row:
            float_values.append(float(column))
            results[float(column)] = material_list[counter] 
            counter += 1

    sorted_float_values = sorted(float_values, reverse=True)
    # print(sorted_float_values)

    index = -1
    for i in range(len(material_list)):
        if material_list[i] == material_input:
            index = i
            break
    if index == -1:
        material_accuracy = None
    else:
        material_accuracy = material_input + ": " + str(float_values[index])

    return [[results[sorted_float_values[0]], sorted_float_values[0]], [results[sorted_float_values[1]], sorted_float_values[1]], [results[sorted_float_values[2]], sorted_float_values[2]], material_accuracy]

inputs = [gr.inputs.Image(type="pil"), gr.inputs.Dropdown(material_list)]
outputs = [gr.outputs.Textbox(label="Top Result"), gr.outputs.Textbox(label="Second Result"), gr.outputs.Textbox(label="Third Result"), gr.outputs.Textbox(label="Material Accuracy")]

interface = gr.Interface(fn=process_image, inputs=inputs, outputs=outputs)
interface.launch()
