import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def generate_finetuning_prompt(image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,  
                },
                {
                    "type": "text",
                    "text": (
                        "Please extract the following details from the Aadhar card in the image: "
                        "- Name of the person, "
                        "- Gender, "
                        "- Date of Birth (DOB), "
                        "- 12-digit Aadhar number, "
                        "- 16-digit VID number,"
                        "- Issue Date"
                        "Return the data in the following structured format in JSON: "
                        "{"
                        "    'name': 'string', "
                        "    'gender': 'string', "
                        "    'date_of_birth': 'DD/MM/YYYY', "
                        "    '12_digit_number': 'string', "
                        "    '16_digit_number': 'string', "
                        "     'Issue Date': 'DD/MM/YYYY'  "
                        "}"
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text[0]

def save_finetuning_prompt(image_name, prompt):
    txt_file_path = os.path.splitext(image_name)[0] + '_finetuning_prompt.txt'
    with open(txt_file_path, 'w') as file:
        file.write(prompt)

def process_images_for_finetuning(folder_path):
    for image_name in os.listdir(folder_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_name)
            print(f"Processing {image_name}...")

            prompt = generate_finetuning_prompt(image_path)

            save_finetuning_prompt(image_name, prompt)
            
            print(f"fine-tuning prompt for {image_name} saved successfully!")

folder_path = r"E:\VISHWA\1.HypervergeNexus\dataforfluroence\imagesdata"  

process_images_for_finetuning(folder_path)
