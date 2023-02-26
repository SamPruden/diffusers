from diffusers import StableDiffusionPipeline, HookLoggingController, Controller, TControllerParams, ResidualStepPatcher
import torch

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
if torch.cuda.is_available(): pipe = pipe.to("cuda")
else: pipe = pipe.to(torch.float32)

class StupidStepPatcher(ResidualStepPatcher):
    def residual(self, hook: str, sample: torch.FloatTensor):
        return 0.1 * torch.randn_like(sample)

class StupidController(Controller[TControllerParams]):
    def __call__(self, *args: TControllerParams.args, **kwargs: TControllerParams.kwargs):
        return StupidStepPatcher(self)

image = pipe(
    prompt = "A beautiful sunset shimmering over the ocean",
    num_inference_steps = 1,
    generator = torch.Generator(device="cpu").manual_seed(0),
    controller = HookLoggingController(),
).images[0]

image.save("demo_output.png")