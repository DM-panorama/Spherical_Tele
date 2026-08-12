import torch
import os
import glob
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig



class ImageStyleInference:
   
    def __init__(self,):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_models()
    
    def _load_models(self):

          model_dir = "/root/autodl-tmp/Qwen-Image-Edit-2509"

          self.pipe = QwenImagePipeline.from_pretrained(
              torch_dtype=torch.bfloat16,
              device="cuda",
              model_configs=[
                  ModelConfig(
                      path=glob.glob(
                          os.path.join(
                              model_dir,
                              "transformer/diffusion_pytorch_model*.safetensors",
                          )
                      )
                  ),
                  ModelConfig(
                      path=glob.glob(
                          os.path.join(
                              model_dir,
                              "text_encoder/model*.safetensors",
                          )
                      )
                  ),
                  ModelConfig(
                      path=os.path.join(
                          model_dir,
                          "vae/diffusion_pytorch_model.safetensors",
                      )
                  ),
              ],
              tokenizer_config=None,
              processor_config=ModelConfig(
                  path=os.path.join(model_dir, "processor")
              ),
          )

          telestyle_image = (
              "weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors"
          )
          speedup = (
              "weights/"
              "diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
          )

          self.pipe.load_lora(self.pipe.dit, telestyle_image)
          self.pipe.load_lora(self.pipe.dit, speedup)

    def inference(self,
        prompt,
        content_ref,
        style_ref,
        seed=123,
        num_inference_steps=4,
        minedge=1024,
        ):
        w, h = Image.open(content_ref).convert("RGB").size
        minedge=minedge-minedge%16

        if w > h:
            r = w / h
            h = minedge
            w = int(h * r) - int(h * r) % 16
        else:
            r = h / w
            w = minedge
            h = int(w * r) - int(w * r) % 16

        images = [
            Image.open(content_ref).convert("RGB").resize((w, h)),
            Image.open(style_ref).convert("RGB").resize((minedge, minedge)),
        ]

        image = self.pipe(
            prompt, 
            edit_image=images, 
            seed=seed, 
            num_inference_steps=num_inference_steps, 
            height=h, 
            width=w,
            edit_image_auto_resize=False,
            cfg_scale=1.0
        )  # lightning

        return image




if __name__ == "__main__":
    inference_engine = ImageStyleInference()

    prompt = 'Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.'
        
    content_ref = "inputs/content.png"
    style_ref = "inputs/style.jpg"
    
    with torch.no_grad():
        generated_image = inference_engine.inference(prompt, content_ref, style_ref, seed=123, num_inference_steps=4, minedge=1024)

    save_dir=f'./qwen_style_output/'

    os.makedirs(save_dir,exist_ok=True)
    prefix=style_ref.split('/')[-1].split('.')[0]


    generated_image.save(os.path.join(save_dir, f'{prefix}_result.png'))


    print(f"saved to {os.path.join(save_dir, f'{prefix}_result.png')}")
            
