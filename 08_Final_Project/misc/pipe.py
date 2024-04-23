import numpy as np
from torch import cuda, device
from transformers import pipeline
from huggingface_hub import InferenceClient

from requests import get
from PIL import Image

from freeGPT import Client
import translators as ts

from misc.labels import labels

import dotenv, os
dotenv.load_dotenv()


class AnsweringPipe():
    models = {i: j for i, j in enumerate(labels['models'])}
    tasks = labels['tasks_prompts']

    def __init__(self, scene_step: int=100, image_max_size: int=1280) -> None:
        self.scene_step = scene_step
        self.image_max_size = image_max_size
        self.device = device('cuda' if cuda.is_available() else 'cpu')

        self.client = InferenceClient(token=os.getenv('HF_KEY'), model='mistralai/Mistral-7B-Instruct-v0.2')
        self.pipe_image_to_text = pipeline('image-to-text', model='microsoft/git-large-coco', device=self.device)
        self.pipe_image_segmentation = pipeline('image-segmentation', model='microsoft/beit-large-finetuned-ade-640-640', device=self.device)

    def preprocess(self, image_link: str) -> Image:
        image = Image.open(get(image_link, stream=True).raw)
        scale = self.image_max_size / max(image.size)
        new_size = tuple([int(x*scale) for x in image.size])
        image = image.resize(new_size, Image.LANCZOS)
        return image
    
    def generate_seg_descr(self, image: Image) -> tuple:
        image_segmentation = self.pipe_image_segmentation.predict(image)
        image_description = self.pipe_image_to_text(image)
        return image_segmentation, image_description
    
    def generate_scene(self, image: Image, image_segmentation: list) -> str:
        image_array = np.array(image)
        image_text = np.empty(image_array.shape[:2], dtype='<U100')

        for mask in image_segmentation:
            mask_array = np.array(mask['mask'])
            image_text[mask_array == 255] = mask['label']
        
        image_text_crop = image_text[::self.scene_step, ::self.scene_step]

        scene = '\n'.join([' | '.join(list(i)) for i in image_text_crop])

        return scene
    
    def generate_selector(self, prompt: str, model_name: str) -> str:
        output_text = None
        if model_name == self.models[0]:
            output_text = self.generate_text_gpt(prompt)
        elif model_name == self.models[1]:
            output_text = self.generate_text_mistral(prompt)
        
        return output_text
    
    def generate_text_gpt(self, prompt: str) -> str:
        output_text = Client.create_completion('gpt3', prompt)

        return self.translate(output_text)
    
    def generate_text_mistral(self, prompt: str) -> str:
        output = self.client.text_generation(
            prompt,
            max_new_tokens=2048,
            return_full_text=False,
        )

        return self.translate(output.strip())
    
    def load_prompt(self, file: str, scene: str, context: str) -> str:
        with open('./prompts/' + file) as f:
            prompt_raw = ''.join(f.readlines())
        return prompt_raw.replace('<SCENE>', scene).replace('<CONTEXT>', context)
    
    def translate(self, text: str, src_lang: str='en', dest_lang: str='ru', translator: str='caiyun') -> str:
        # alternative option for translator is 'modernMt', it's also good
        return ts.translate_text(text, translator=translator, from_language=src_lang, to_language=dest_lang)
    
    def predict(self, image_link: str, model_name: str=models[0], task: int=0) -> str:
        image = self.preprocess(image_link)
        image_segmentation, image_description = self.generate_seg_descr(image)

        scene = self.generate_scene(image, image_segmentation)
        context = image_description[0].get('generated_text')
        
        prompt = self.load_prompt(self.tasks[task], scene, context)
        output = self.generate_selector(prompt, model_name=model_name)

        return output

    def __call__(self, *args: np.any, **kwds: np.any) -> str:
        return self.predict(*args, **kwds)


ap = AnsweringPipe()
def answer_image(image_link, model_name, task):
    output = ap(image_link, model_name, task)
    return output