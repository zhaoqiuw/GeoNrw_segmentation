import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models.fcn import FCN
import segmentation_models_pytorch as smp

from optimizer import update_ema_variables


def softmax_mse_loss(preds, targets, ignore_index=0):
    preds = preds.permute(0, 2, 3, 1)
    preds = torch.sigmoid(preds)

    targets = targets.permute(0, 2, 3, 1)
    targets = torch.sigmoid(targets)

    if ignore_index is not None:
        mask = torch.ones(targets.shape[-1])
        mask[ignore_index] = 0.
        mask = mask.type(torch.bool)

        preds = preds[:, :, :, mask]
        targets = targets[:, :, :, mask]

    return F.mse_loss(preds, targets, reduction='mean')


class SemiSegNet(nn.Module):
    def __init__(self, model_type='deeplabv3+', encoder_name='resnet18', encoder_weights='imagenet',
                 in_channels=4, num_filters=16, num_classes=16, ignore_index=0, dice_weight=1.0,
                 consistency_weight=0.1, consistency_ema_weight=0.1, consistency_filp_weight=0.1,
                 consistency_filp_ema_weight=0.1):
        super(SemiSegNet, self).__init__()

        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.consistency_weight = consistency_weight
        self.consistency_ema_weight = consistency_ema_weight
        self.consistency_filp_weight = consistency_filp_weight
        self.consistency_filp_ema_weight = consistency_filp_ema_weight

        if model_type == 'unet':
            self.model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                  in_channels=in_channels, classes=num_classes)
            self.model_ema = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                      in_channels=in_channels, classes=num_classes)
        elif model_type == 'deeplabv3+':
            self.model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                           in_channels=in_channels, classes=num_classes)
            self.model_ema = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                               in_channels=in_channels, classes=num_classes)
        elif model_type == 'fcn':
            self.model = FCN(in_channels=in_channels, classes=num_classes, num_filters=num_filters)
            self.model_ema = FCN(in_channels=in_channels, classes=num_classes, num_filters=num_filters)
        else:
            raise ValueError(f"Model type '{model_type}' is not valid.")
        # mean teacher 使用
        # for param in self.model_ema.parameters():
        #     param.detach_()

        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', ignore_index=ignore_index, normalized=False)
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)

    def forward_cps(self, label_batch, unlabel_batch, global_step, consistency=1.0):
        #对输入的label_batch不用做修改
        pred_sup_l = self.model(label_batch['image'])
        pred_unsup_l= self.model(unlabel_batch['image'])
        pred_sup_r = self.model_ema(label_batch['image'])
        pred_unsup_r = self.model_ema(unlabel_batch['image'])
        ### cps loss ###
        pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
        pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
        _, max_l = torch.max(pred_l, dim=1)
        _, max_r = torch.max(pred_r, dim=1)
        max_l = max_l.long()
        max_r = max_r.long()
        cps_loss = self.criterion(pred_l, max_r) + self.criterion(pred_r, max_l)
        ### standard cross entropy loss ###
        loss_sup_l = self.criterion(pred_sup_l, label_batch['mask'])
        loss_sup_r = self.criterion(pred_sup_r, label_batch['mask'])
        loss_all = cps_loss + loss_sup_l + loss_sup_r
        loss = {
            'loss_all': loss_all, 'cps_loss': cps_loss, 'loss_sup_l': loss_sup_l,
            'logit': pred_sup_l, 'label': label_batch['mask'],
        }
        return loss




    def forward_semi(self, label_batch, unlabel_batch, global_step, consistency=1.0):
        update_ema_variables(self.model, self.model_ema, 0.999, float(global_step / 100))

        patch_size = label_batch['image'].shape[-1] // 2
        label_batch1 = {'image': label_batch['image'][:, :, :patch_size * 2, :],
                        'mask': label_batch['mask'][:, :patch_size * 2, :]}
        label_batch2 = {'image': label_batch['image'][:, :, patch_size:, :],
                        'mask': label_batch['mask'][:, patch_size:, :]}
        unlabel_batch1 = {'image': unlabel_batch['image'][:, :, :patch_size * 2, :]}
        unlabel_batch2 = {'image': unlabel_batch['image'][:, :, patch_size:, :]}

        label_student_logit1 = self.model(label_batch1['image'])
        unlabel_student_logit1 = self.model(unlabel_batch1['image'])
        with torch.no_grad():
            label_teacher_logit1 = self.model_ema(label_batch1['image'])
            label_teacher_logit2 = self.model_ema(label_batch2['image'])
            unlabel_teacher_logit1 = self.model_ema(unlabel_batch1['image'])
            unlabel_teacher_logit2 = self.model_ema(unlabel_batch2['image'])

        batch_focal_loss = self.focal_loss(label_student_logit1, label_batch1['mask'])
        batch_dice_loss = self.dice_loss(label_student_logit1, label_batch1['mask'])

        batch_consistency_loss = consistency * softmax_mse_loss(label_student_logit1, label_teacher_logit1,
                                                                ignore_index=self.ignore_index)
        batch_consistency_loss_ema = consistency * softmax_mse_loss(unlabel_student_logit1, unlabel_teacher_logit1,
                                                                    ignore_index=self.ignore_index)

        batch_consistency_flip_loss = consistency * softmax_mse_loss(label_student_logit1[:, :, patch_size:, :],
                                                                     label_teacher_logit2[:, :, :patch_size, :],
                                                                     ignore_index=self.ignore_index)
        batch_consistency_flip_loss_ema = consistency * softmax_mse_loss(unlabel_student_logit1[:, :, patch_size:, :],
                                                                         unlabel_teacher_logit2[:, :, :patch_size, :],
                                                                         ignore_index=self.ignore_index)

        loss_all = batch_focal_loss + self.dice_weight * batch_dice_loss + \
                   self.consistency_weight * batch_consistency_loss + \
                   self.consistency_ema_weight * batch_consistency_loss_ema + \
                   self.consistency_filp_weight * batch_consistency_flip_loss + \
                   self.consistency_filp_ema_weight * batch_consistency_flip_loss_ema

        loss = {
            'loss_all': loss_all, 'focal_loss': batch_focal_loss, 'dice_loss': batch_dice_loss,
            'consistency_loss': batch_consistency_loss, 'consistency_loss_ema': batch_consistency_loss_ema,
            'consistency_flip_loss': batch_consistency_flip_loss,
            'consistency_flip_loss_ema': batch_consistency_flip_loss_ema,
            'logit': label_student_logit1, 'label': label_batch1['mask'],
        }
        return loss

    def forward_sup(self, batch):
        logits = self.model(batch['image'])
        batch_focal_loss = self.focal_loss(logits, batch['mask'])
        batch_dice_loss = self.dice_loss(logits, batch['mask'])

        loss_all = batch_focal_loss + self.dice_weight * batch_dice_loss
        loss = {
            'logit': logits, 'label': batch['mask'], 'loss_all': loss_all,
            'focal_loss': batch_focal_loss, 'dice_loss': batch_dice_loss,
        }
        return loss

    def forward(self, tensor_dict, semi=True):
        if semi:
            return self.forward_cps(tensor_dict['label_batch'], tensor_dict['unlabel_batch'],
                                     tensor_dict['global_step'],
                                     consistency=tensor_dict['consistency'])
        else:
            return self.forward_sup(tensor_dict['batch'])
