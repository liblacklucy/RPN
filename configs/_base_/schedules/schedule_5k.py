# optimizer
optimizer = dict(type='SGD', lr=0.01, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=5000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=500, metric='mIoU', save_best='U_mIoU')
 