_base_ = './voc12_20_512x512.py'
# dataset settings
data = dict(
    train=dict(
        ann_dir='SegmentationClassAug',
        split='list/train_aug.txt',
        ),
    val=dict(
        ann_dir='SegmentationClassAug',
        split='list/val.txt',
        ),
    test=dict(
        ann_dir='SegmentationClassAug',
        split='list/val.txt',
        )
    )