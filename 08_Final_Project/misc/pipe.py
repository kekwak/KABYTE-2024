import numpy as np
import random

from torch import cuda, device
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import InferenceClient

# from importlib import reload
from requests import get
from PIL import Image
from tqdm import tqdm

from g4f.client import Client
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
        self.img_tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-1_8b', trust_remote_code=True)
        self.visual_model = AutoModelForCausalLM.from_pretrained('internlm/internlm-xcomposer2-vl-1_8b', trust_remote_code=True).cuda().eval()
        self.pipe_image_to_text = pipeline('image-to-text', model='microsoft/git-large-coco', device='cpu')
        self.pipe_image_segmentation = pipeline('image-segmentation', model='microsoft/beit-large-finetuned-ade-640-640', device='cpu')

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
    
    def generate_descr_advanced(self, image: Image) -> str:
        questions = labels['vqa_questions']

        file_id = str(random.randint(1e+20, 1e+21))
        file_path = f'images_tmp/{file_id}.png'
        image.save(file_path)

        model_answers = []
        for question in tqdm(questions):
            model_question = '<ImageHere>' + question
            with cuda.amp.autocast():
                out_final, _ = self.visual_model.chat(self.img_tokenizer, query=model_question, image=file_path, do_sample=False, history=[])
            
            model_answers.append(out_final)

        os.remove(file_path)
        output = '\n\n'.join(model_answers)

        return output
    
    def answer_question_advanced(self, image: Image, question: str) -> str:
        file_id = str(random.randint(1e+20, 1e+21))
        file_path = f'images_tmp/{file_id}.png'
        image.save(file_path)

        model_question = '<ImageHere>' + question
        for _ in tqdm(range(1)):
            with cuda.amp.autocast():
                output, _ = self.visual_model.chat(self.img_tokenizer, query=model_question, image=file_path, do_sample=False, history=[])

        os.remove(file_path)

        return output
    
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
        
        return self.translate(output_text)
    
    def generate_text_gpt(self, prompt: str) -> str:
        client = Client()
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': prompt}],
        )

        return response.choices[0].message.content
    
    def generate_text_mistral(self, prompt: str) -> str:
        output = self.client.text_generation(
            prompt,
            max_new_tokens=1024,
            return_full_text=False,
        )

        return output.strip()
    
    def load_prompt(self, file: str, scene: str, context: str) -> str:
        with open('./prompts/' + file) as f:
            prompt_raw = ''.join(f.readlines())
        return prompt_raw.replace('<SCENE>', scene).replace('<CONTEXT>', context)
    
    def translate(self, text: str, src_lang: str='en', dest_lang: str='ru', translator: str='modernMt', max_length: int=4096) -> str:
        text_parts = []
        for start in range(0, len(text), max_length):
            end = start + max_length
            try:
                # reload(ts)  # translators issue fix
                # alternative option for translator is 'modernMt', it's also good ('caiyun' now)
                out = ts.translate_text(
                    text[start:end], translator=translator, from_language=src_lang, to_language=dest_lang,
                    # update_session_after_freq=1, update_session_after_seconds=5,
                )
            except:
                out = labels['error']
            text_parts.append(out)
        
        return ' '.join(text_parts)
    
    def predict(self, image_link: str, model_name: str=models[0], task: int=0, advanced: bool=True, question: str='') -> str:
        prompt_file = self.tasks[task]
        image = self.preprocess(image_link)
        if task == 2:
            in_text = self.translate(question, src_lang='auto', dest_lang='en')
            out_text = self.answer_question_advanced(image, in_text)
            output = self.translate(out_text, src_lang='en', dest_lang='ru', translator='bing')
        elif not advanced:
            image_segmentation, image_description = self.generate_seg_descr(image)

            context = image_description[0].get('generated_text')
            scene = self.generate_scene(image, image_segmentation)
        else:
            prompt_file_sp = prompt_file.split('.')
            prompt_file = f'{prompt_file_sp[0]}_advanced.{prompt_file_sp[1]}'

            context = self.generate_descr_advanced(image)
            scene=''
        
        if task != 2:
            prompt = self.load_prompt(prompt_file, scene, context)
            output = self.generate_selector(prompt, model_name=model_name)

        return output

    def __call__(self, *args: np.any, **kwds: np.any) -> str:
        return self.predict(*args, **kwds)


ap = AnsweringPipe()
def answer_image(*args, **kwargs):
    output = ap(*args, **kwargs)
    return output