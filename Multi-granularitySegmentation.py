import os
import torch
from semantic_sam import prepare_image, plot_multi_results, build_semantic_sam, SemanticSAMPredictor

# set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'

torch.cuda.empty_cache()

original_image, input_image = prepare_image(image_pth='/home/keshavkumartum/ssam11feb/Semantic-SAM/input.png')
mask_generator = SemanticSAMPredictor(build_semantic_sam(model_type='L', ckpt='/home/keshavkumartum/SEMANTIC-SAM-datasets/COCO/DATASETS/CheckPoints from ModelZoo/swinl_only_sam_many2many.pth'))

# Use torch.no_grad() for the inference part
with torch.no_grad():
    iou_sort_masks, area_sort_masks = mask_generator.predict_masks(original_image, input_image, point=[[0.5, 0.5]]) 

plot_multi_results(iou_sort_masks, area_sort_masks, original_image, save_path='../vis/')
