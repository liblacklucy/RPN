# VOC2012
#CUDA_VISIBLE_DEVICES="6" python test.py configs/voc12/CrossPromptCLIP_seg_zero_vit-b_512x512_20k_12_10_debug2.py YOUR PTH --eval=mIoU

# COCO
CUDA_VISIBLE_DEVICES="2" python test.py configs/coco/CrossPromptCLIP_seg_zero_vit-b_512x512_80k_12_100_multi_debug8.py YOUR PTH --eval=mIoU
