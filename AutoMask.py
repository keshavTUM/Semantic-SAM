import os
import torch
import torch.nn.functional as F
from semantic_sam import prepare_image, plot_results, build_semantic_sam, SemanticSamAutomaticMaskGenerator
from functools import wraps
from torchvision import transforms
import numpy as np

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:500'

torch.cuda.empty_cache()

# Prepare the image and resize it to a smaller resolution
resize_transform = transforms.Resize((512, 512))
original_image, input_image = prepare_image(image_pth='/home/keshavkumartum/ssam11feb/Semantic-SAM/input.png')
resized_image = resize_transform(input_image)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
semantic_sam_model = build_semantic_sam(model_type='L', ckpt='/home/keshavkumartum/SEMANTIC-SAM-datasets/COCO/DATASETS/CheckPoints from ModelZoo/swinl_only_sam_many2many.pth').to(device)

# Initialize the mask generator with specified level for granularity
mask_generator = SemanticSamAutomaticMaskGenerator(semantic_sam_model, level=[1, 2, 3, 4, 5, 6])  # Use all six levels by default

def retry_if_cuda_oom(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory. " not in str(e):
                raise
        # Clear cache and retry
        torch.cuda.empty_cache()
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory. " not in str(e):
                raise

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        print(f"Attempting to copy inputs of {str(func)} to CPU due to CUDA OOM")
        new_args = tuple(x.to(device="cpu") if hasattr(x, "to") and x.device.type == "cuda" else x for x in args)
        new_kwargs = {k: (v.to(device="cpu") if hasattr(v, "to") and v.device.type == "cuda" else v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped

# Use torch.no_grad() for the inference part
with torch.no_grad():
    # Move the resized input image to the same device as the model
    resized_image = resized_image.to(device)
    
    # Generate masks with the resized image
    masks = retry_if_cuda_oom(mask_generator.generate)(resized_image)
    
    # Resize masks to the original image dimensions
    masks_resized = []
    for mask in masks:
        mask_resized = {
            key: F.interpolate(
                torch.tensor(value).unsqueeze(0).unsqueeze(0).float(),
                size=original_image.shape[1:3],  # H, W
                mode='nearest'
            ).squeeze(0).squeeze(0).numpy() 
            for key, value in mask.items() if isinstance(value, np.ndarray) and value.ndim == 2  # Ensure it's 2D
        }
        masks_resized.append(mask_resized)

# Plot the results
plot_results(masks_resized, original_image, save_path='../vis/')