from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler, AutoencoderTiny
from main.sdxl.sdxl_text_encoder import SDXLTextEncoder
from main.utils import get_x0_from_noise
from transformers import AutoTokenizer
from accelerate import Accelerator
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import numpy as np
import argparse 
import torch
import time 
import cv2
from safety_checker.censor import check_safety
import io
import uvicorn
import base64

SAFETY_CHECKER = True

# Global variables to track time
total_request_time_accumulated = 0
first_request_time = None
request_count = 0

class ModelWrapper:
    def __init__(self, args, accelerator):
        super().__init__()
        # disable all gradient calculations
        torch.set_grad_enabled(False)
        
        if args.precision == "bfloat16":
            self.DTYPE = torch.bfloat16
        elif args.precision == "float16":
            self.DTYPE = torch.float16
        else:
            self.DTYPE = torch.float32
        self.device = accelerator.device

        self.tokenizer_one = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
        )

        self.tokenizer_two = AutoTokenizer.from_pretrained(
            args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
        )

        self.text_encoder = SDXLTextEncoder(args, accelerator).to(dtype=self.DTYPE)

        # Initialize AutoEncoder with specified model and dtype
        if args.use_tiny_vae:
            self.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl", 
                torch_dtype=self.DTYPE
            ).to(self.device)
        else:
            self.vae = AutoencoderKL.from_pretrained(
                args.model_id, 
                subfolder="vae"
            ).to(self.device).float()

        # Initialize Generator
        self.model = self.create_generator(args).to(dtype=self.DTYPE).to(self.device)

        self.accelerator = accelerator
        self.latent_resolution = args.latent_resolution
        self.num_train_timesteps = args.num_train_timesteps

        self.conditioning_timestep = args.conditioning_timestep 

        self.scheduler = DDIMScheduler.from_pretrained(
            args.model_id,
            subfolder="scheduler"
        )
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        # sampling parameters 
        self.num_step = args.num_step 
        self.conditioning_timestep = args.conditioning_timestep 


    def create_generator(self, args):
        generator = UNet2DConditionModel.from_pretrained(
            args.model_id,
            subfolder="unet"
        ).to(self.DTYPE)

        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        print(generator.load_state_dict(state_dict, strict=True))
        generator.requires_grad_(False)
        return generator 

    def build_condition_input(self, width, height):
        original_size = (width, height)
        target_size = (width, height)
        crop_top_left = (0, 0)

        add_time_ids = list(original_size + crop_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], device=self.device, dtype=self.DTYPE)
        return add_time_ids

    def _encode_prompts(self, prompts):
        text_input_ids_one = self.tokenizer_one(
            prompts,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        text_input_ids_two = self.tokenizer_two(
            prompts,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        prompt_dict = {
            'text_input_ids_one': text_input_ids_one.to(self.device),
            'text_input_ids_two': text_input_ids_two.to(self.device)
        }
        return prompt_dict 
    
    @staticmethod
    def _get_time():
        torch.cuda.synchronize()
        return time.time()

    def sample(
        self, noise, unet_added_conditions, prompt_embed
    ):
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        if self.num_step == 1:
            all_timesteps = [self.conditioning_timestep]
            step_interval = 0 
        elif self.num_step == 4:
            all_timesteps = [999, 749, 499, 249]
            step_interval = 250 
        else:
            raise NotImplementedError()
        
        DTYPE = prompt_embed.dtype
        
        for constant in all_timesteps:
            current_timesteps = torch.ones(len(prompt_embed), device=self.device, dtype=torch.long)  *constant
            eval_images = self.model(
                noise, current_timesteps, prompt_embed, added_cond_kwargs=unet_added_conditions
            ).sample

            eval_images = get_x0_from_noise(
                noise, eval_images, alphas_cumprod, current_timesteps
            ).to(self.DTYPE)

            next_timestep = current_timesteps - step_interval 
            noise = self.scheduler.add_noise(
                eval_images, torch.randn_like(eval_images), next_timestep
            ).to(DTYPE)  

        eval_images = self.vae.decode(eval_images / self.vae.config.scaling_factor, return_dict=False)[0]
        eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        return eval_images 

    @torch.no_grad()
    def inference(
        self,
        prompts: list,
        seed: int,
        width: int,
        height: int
    ):
        num_images = len(prompts)
        print(f"Running model inference... images: {num_images}")

        if seed == -1:
            seed = np.random.randint(0, 1000000)

        generator = torch.manual_seed(seed)

        base_add_time_ids = self.build_condition_input(width, height)

        add_time_ids = base_add_time_ids.repeat(num_images, 1)

        noise = torch.randn(
            num_images, 4, height // 8, width // 8, 
            generator=generator
        ).to(device=self.device, dtype=self.DTYPE) 

        # Log the start time for encoding prompts
        encode_start_time = self._get_time()
        prompt_inputs = self._encode_prompts(prompts)
        encode_end_time = self._get_time()
        print(f"Encoding prompts took {(encode_end_time - encode_start_time):.2f} seconds")

        # Log the start time for text encoding
        text_encode_start_time = self._get_time()
        batch_prompt_embeds, batch_pooled_prompt_embeds = self.text_encoder(prompt_inputs)
        text_encode_end_time = self._get_time()
        print(f"Text encoding took {(text_encode_end_time - text_encode_start_time):.2f} seconds")

        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": batch_pooled_prompt_embeds.squeeze(1)
        }

        # Log the start time for sampling
        sample_start_time = self._get_time()
        eval_images = self.sample(
            noise=noise,
            unet_added_conditions=unet_added_conditions,
            prompt_embed=batch_prompt_embeds
        )
        sample_end_time = self._get_time()
        print(f"Sampling images took {(sample_end_time - sample_start_time):.2f} seconds")

        output_image_list = [] 
        for image in eval_images:
            image_np = image.cpu().numpy()
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            output_image_list.append(image_bgr)

        # Log the total inference time
        total_inference_time = sample_end_time - encode_start_time
        print(f"Total inference time: {total_inference_time:.2f} seconds")

        return output_image_list, eval_images

app = FastAPI()

# Initialize model once at startup
args = argparse.Namespace(
    latent_resolution=128,
    num_train_timesteps=1000,
    checkpoint_path='./sdxl_cond999.bin',
    model_id='stabilityai/stable-diffusion-xl-base-1.0',
    precision='float16',
    use_tiny_vae=True,
    conditioning_timestep=999,
    num_step=4,
    revision=None
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True 

accelerator = Accelerator()
model = ModelWrapper(args, accelerator)

@app.post('/generate')
async def generate(request: Request):
    global total_request_time_accumulated, first_request_time, request_count

    data = await request.json()
    prompts = data.get('prompts', ['children'])

    def convert_to_int(value, default):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    width = convert_to_int(data.get('width', 1024), 1024)
    height = convert_to_int(data.get('height', 1024), 1024)
    seed = convert_to_int(data.get('seed', -1), -1)

    # Log the start time for the entire request processing
    request_start_time = time.time()

    # Generate images for each prompt
    all_images, all_images_tensor = model.inference(prompts, seed, width, height)

    if not all_images:
        return JSONResponse(content={"error": "No images generated"}, status_code=500)

    response_content = []
    total_image_creation_time = 0
    total_safety_check_time = 0

    # Log the start time for image creation
    image_creation_start_time = time.time()

    # Process images in batch
    img_byte_arr_list = []
    for image in all_images:
        img_byte_arr = io.BytesIO()
        _, img_encoded = cv2.imencode('.png', image)
        img_byte_arr.write(img_encoded.tobytes())
        img_byte_arr_list.append(img_byte_arr.getvalue())

    # Convert images to base64
    img_base64_list = [base64.b64encode(img_byte_arr).decode('utf-8') for img_byte_arr in img_byte_arr_list]

    # Log the end time for image creation
    image_creation_end_time = time.time()
    image_creation_time = image_creation_end_time - image_creation_start_time
    total_image_creation_time += image_creation_time
    print(f"Image creation time: {image_creation_time:.2f} seconds")

    # Log the start time for the safety checker
    safety_check_start_time = time.time()
    print("starting safety check")
    concepts, has_nsfw_concepts_list = check_safety(all_images_tensor, safety_checker_adj=0.0)
    print("end safety check")
    # Log the end time for the safety checker
    safety_check_end_time = time.time()
    safety_check_time = safety_check_end_time - safety_check_start_time
    total_safety_check_time += safety_check_time
    print(f"Safety check time: {safety_check_time:.2f} seconds")

    for img_base64, prompt, has_nsfw_concept, concept in zip(img_base64_list, prompts, has_nsfw_concepts_list, concepts):
        image_content = {
            "image": img_base64,
            "has_nsfw_concept": has_nsfw_concept,
            "concept": concept,
            "width": width,
            "height": height,
            "seed": seed,
            "prompt": prompt
        }

        response_content.append(image_content)

    # Log the end time for the entire request processing
    request_end_time = time.time()
    total_request_time = request_end_time - request_start_time

    # Update global time accumulators and request count
    total_request_time_accumulated += total_request_time
    request_count += 1

    # Record the time of the first request
    if first_request_time is None:
        first_request_time = request_start_time

    # Calculate the total time passed since the first request
    total_time_passed = request_end_time - first_request_time

    # Calculate the percentage of time spent processing requests
    percentage_time_processing = (total_request_time_accumulated / total_time_passed) * 100

    print(f"Total request time: {total_request_time:.2f} seconds")
    print(f"Percentage of time spent processing requests: {percentage_time_processing:.2f}%")

    return JSONResponse(content=response_content, media_type="application/json")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=5000)
