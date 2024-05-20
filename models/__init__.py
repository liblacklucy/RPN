from models.segmentor.cptclip import CrossPromptCLIP

from models.backbone.text_encoder import CPTCLIPTextEncoder
from models.backbone.img_encoder import CPTCLIPVisionTransformer
from models.decode_heads.decode_seg import CLIPHeadSeg

from models.losses.atm_loss import SegLossPlus  # Sigmoid LOSS
from models.losses.mpt_loss import SegLoss  # Softmax LOSS

from configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
from configs._base_.datasets.dataloader.coco_stuff import ZeroCOCOStuffDataset