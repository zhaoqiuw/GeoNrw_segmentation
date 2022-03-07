import os
import sys
import time
import yaml
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchmetrics

from dataset_train import GEONRW_DataSet
from models_cps import SemiSegNet
from optimizer import get_current_consistency_weight



def set_seed_logger(train_cfg):
    seed = train_cfg['SEED']
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(train_cfg['OUT_PATH'], 'train.log')),
                            logging.StreamHandler(sys.stdout)])


def init_model(model_cfg, device):
    model = SemiSegNet(
        model_type=model_cfg['MODEL_TYPE'], encoder_name=model_cfg['ENCODER_NAME'],
        encoder_weights=model_cfg['ENCODER_WEIGHTS'],in_channels=model_cfg['IN_CHANNLES'],
        num_filters=model_cfg['NUM_FILTERS'], num_classes=model_cfg['NUM_CLASSES'],
        ignore_index=model_cfg['IGNORE_INDEX'], dice_weight=model_cfg['DICE_WEIGHT'],
        consistency_weight=model_cfg['CONSISTENCY_WEIGHT'],
        consistency_ema_weight=model_cfg['CONSISTENCY_EMA_WEIGHT'],
        consistency_filp_weight=model_cfg['CONSISTENCY_FLIP_WEIGHT'],
        consistency_filp_ema_weight=model_cfg['CONSISTENCY_FLIP_EMA_WEIGHT']
    )
    for param in model.model_ema.parameters():
        param.detach_()

    if os.path.isfile(model_cfg['CHECKPOINT']):
        state = torch.load(model_cfg['CHECKPOINT'], map_location='cpu')['state_dict']
        model.load_state_dict(state)
        logging.info('Load model checkpoint in: %s', model_cfg['CHECKPOINT'])
    model = torch.nn.DataParallel(model).to(device)
    return model


def prep_optimizer(optimizer_cfg, model):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=optimizer_cfg['TRAINING_LR'],
                                 weight_decay=optimizer_cfg['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optimizer_cfg['STEP_SIZE'],
                                                gamma=optimizer_cfg['STEP_GAMMA'])
    return optimizer, scheduler


def get_dataloader(dataset_cfg):
    dataset = GEONRW_DataSet(dataset_cfg['ROOT_DIR'], dataset_cfg['TRAIN_BATCH'], dataset_cfg['VAL_BATCH'],
                             dataset_cfg['NUM_WORKERS'], dataset_cfg['VAL_SPLIT_PCT'], dataset_cfg['PATCH_SIZE'],
                             gpu_num=dataset_cfg['GPU_NUM'], shuffle_dataset=dataset_cfg['SHUFFLE_DATASET'])

    train_label_loader = dataset.train_label_dataloader()
    train_unlabel_loader = dataset.train_unlabel_dataloader()
    val_label_loader = dataset.val_label_dataloader()
    return train_label_loader, train_unlabel_loader, val_label_loader


def gen_metrics(mertrics_cfg):
    metrics = {
        'OverallAccuracy': torchmetrics.Accuracy(
            num_classes=mertrics_cfg['NUM_CLASS'],
            average='micro',
            mdmc_average='global',
            ignore_index=mertrics_cfg['IGNORE_INDEX'],
        ),
        'OverallPrecision': torchmetrics.Precision(
            num_classes=mertrics_cfg['NUM_CLASS'],
            average='micro',
            mdmc_average='global',
            ignore_index=mertrics_cfg['IGNORE_INDEX'],
        ),
        'OverallRecall': torchmetrics.Recall(
            num_classes=mertrics_cfg['NUM_CLASS'],
            average='micro',
            mdmc_average='global',
            ignore_index=mertrics_cfg['IGNORE_INDEX'],
        ),
        'AverageAccuracy': torchmetrics.Accuracy(
            num_classes=mertrics_cfg['NUM_CLASS'],
            average='macro',
            mdmc_average='global',
            ignore_index=mertrics_cfg['IGNORE_INDEX'],
        ),
        'AveragePrecision': torchmetrics.Precision(
            num_classes=mertrics_cfg['NUM_CLASS'],
            average='macro',
            mdmc_average='global',
            ignore_index=mertrics_cfg['IGNORE_INDEX'],
        ),
        'AverageRecall': torchmetrics.Recall(
            num_classes=mertrics_cfg['NUM_CLASS'],
            average='macro',
            mdmc_average='global',
            ignore_index=mertrics_cfg['IGNORE_INDEX'],
        ),
        'IoU': torchmetrics.IoU(
            num_classes=mertrics_cfg['NUM_CLASS'],
            ignore_index=mertrics_cfg['IGNORE_INDEX'],
        ),
        'F1Score': torchmetrics.FBeta(
            num_classes=mertrics_cfg['NUM_CLASS'],
            beta=1.0,
            average='micro',
            mdmc_average='global',
            ignore_index=mertrics_cfg['IGNORE_INDEX'],
        ),
    }
    return metrics


def train_semi_epoch(train_cfg, train_label_loader, train_unlabel_loader, model,
                     optimizer, metrics, epoch, global_step, device):
    torch.cuda.empty_cache()
    model.train()

    labeled_trainiter = iter(train_label_loader)
    unlabeled_trainiter = iter(train_unlabel_loader)
    epoch_records = {}
    for k in metrics:
        metrics[k].reset()

    pbar = tqdm(range(len(train_label_loader)))
    start_time = time.time()
    for n_iter in pbar:
        global_step += 1
        pbar.set_description('train semi epoch: {}/{}'.format(epoch + 1, train_cfg['TRAIN_EPOCHS']))
        input_data_label = labeled_trainiter.next()
        try:
            input_data_unlabel = unlabeled_trainiter.next()
        except:
            unlabeled_trainiter = iter(train_unlabel_loader)
            input_data_unlabel = unlabeled_trainiter.next()

        input_data_label = {'image': input_data_label['image'].to(device),
                            'mask': input_data_label['mask'].squeeze(1).to(device)}
        input_data_unlabel = {'image': input_data_unlabel['image'].to(device)}
        tensor_dict = {
            'label_batch': input_data_label, 'unlabel_batch': input_data_unlabel, 'global_step': global_step,
            'consistency': get_current_consistency_weight(epoch, train_cfg['CONSISTENCY'], train_cfg['CONSISTENCY_RAMPUP'])
        }

        batch_result = model(tensor_dict, semi=True)
        optimizer.zero_grad()
        batch_result['loss_all'].mean().backward()
        optimizer.step()

        for k in batch_result.keys():
            if k not in ['logit', 'label']:
                if k not in epoch_records:
                    epoch_records[k] = 0
                epoch_records[k] += batch_result[k].mean().item()

        batch_metric = {}
        for k in metrics.keys():
            batch_metric[k] = metrics[k](batch_result['logit'].detach().cpu(), batch_result['label'].detach().cpu())

        if (n_iter + 1) % train_cfg['LOG_STEP'] == 0:
            current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
            msg_list = []
            for k in batch_result.keys():
                if k not in ['logit', 'label']:
                    msg_list.append('{}: {:.05f}'.format(k, batch_result[k].mean().item()))
            for k in metrics.keys():
                msg_list.append('{}: {:.05f}'.format(k, batch_metric[k]))
            msg = '  Train SEMI | epoch: {:d}/{:d}, step: {:d}/{:d}, global step: {:d}, lr: {:.05f}, time/step: {:.05f} \n'.format(
                epoch + 1, train_cfg['TRAIN_EPOCHS'], n_iter + 1, len(train_label_loader), global_step,
                current_lr, (time.time() - start_time) / train_cfg['LOG_STEP']
            )
            for k in range(len(msg_list)):
                if k % 3 == 0:
                    msg += '     ' + msg_list[k]
                else:
                    msg += ', ' + msg_list[k]
                if k % 3 == 2 and k < len(msg_list) - 1:
                    msg += ' \n'
            logging.info(msg)
            start_time = time.time()

    for k in epoch_records.keys():
        epoch_records[k] = epoch_records[k] / len(train_label_loader)
    for k in metrics.keys():
        epoch_records[k] = metrics[k].compute()
    msg_list = []
    for k in epoch_records.keys():
        msg_list.append('{}: {:.05f}'.format(k, epoch_records[k]))

    msg = 'Train Semi Finish epoch: {:d}/{:d} \n'.format(epoch + 1, train_cfg['TRAIN_EPOCHS'])
    for k in range(len(msg_list)):
        if k % 3 == 0:
            msg += '     ' + msg_list[k]
        else:
            msg += ', ' + msg_list[k]
        if k % 3 == 2 and k < len(msg_list) - 1:
            msg += ' \n'
    logging.info(msg)

    return epoch_records


def val_epoch(train_cfg, dataloader, model, metrics, epoch, device):
    torch.cuda.empty_cache()
    model.eval()
    for k in metrics:
        metrics[k].reset()

    epoch_records = {}
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for n_iter, batch in pbar:
            pbar.set_description('val epoch: {}/{}'.format(epoch + 1, train_cfg['TRAIN_EPOCHS']))

            batch = {'image': batch['image'].to(device), 
                     'mask': batch['mask'].squeeze(1).to(device)}
            tensor_dict = {'batch': batch}
            batch_result = model(tensor_dict, semi=False)

            for k in batch_result.keys():
                if k not in ['logit', 'label']:
                    if k not in epoch_records:
                        epoch_records[k] = 0
                    epoch_records[k] += batch_result[k].mean().item()
            for k in metrics.keys():
                metrics[k](batch_result['logit'].detach().cpu(), batch_result['label'].detach().cpu())

    for k in epoch_records.keys():
        epoch_records[k] = epoch_records[k] / len(dataloader)
    for k in metrics.keys():
        epoch_records[k] = metrics[k].compute()
    msg_list = []
    for k in epoch_records.keys():
        msg_list.append('{}: {:.05f}'.format(k, epoch_records[k]))

    msg = 'Val Finish epoch: {:d}/{:d} \n'.format(epoch + 1, train_cfg['TRAIN_EPOCHS'])
    for k in range(len(msg_list)):
        if k % 3 == 0:
            msg += '     ' + msg_list[k]
        else:
            msg += ', ' + msg_list[k]
        if k % 3 == 2 and k < len(msg_list) - 1:
            msg += ' \n'
    logging.info(msg)
    return epoch_records


def train(model_cfg, optimizer_cfg, dataset_cfg, train_cfg, mertrics_cfg, device):
    set_seed_logger(train_cfg)
    output_folder = os.path.join(train_cfg['OUT_PATH'], 'models')
    os.makedirs(output_folder, exist_ok=True)

    model = init_model(model_cfg, device)
    optimizer, scheduler = prep_optimizer(optimizer_cfg, model)

    train_label_loader, train_unlabel_loader, val_label_loader = get_dataloader(dataset_cfg)
    metrics = gen_metrics(mertrics_cfg)

    logging.info('Model = %s', str(model))
    logging.info('Model config = %s', str(dict(model_cfg)))
    logging.info('Dataset config = %s', str(dict(dataset_cfg)))
    logging.info('Optimizer config = %s', str(dict(optimizer_cfg)))
    logging.info('Train config = %s', str(dict(train_cfg)))
    logging.info('metrics config = %s', str(dict(mertrics_cfg)))
    logging.info('Train parameters = %d', sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info('Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('Train label loader len = %d', len(train_label_loader))
    logging.info('Train unlabel loader len = %d', len(train_unlabel_loader))
    logging.info('val label loader len = %d', len(val_label_loader))

    records = {'epoch': []}
    global_step = 0
    best_IoU = 0.
    for epoch in range(train_cfg['TRAIN_EPOCHS']):
        logging.info(('=' * 50) + 'EPOCH: ' + str(epoch + 1) + ('=' * 50))
        records['epoch'].append(epoch + 1)

        train_result = train_semi_epoch(train_cfg, train_label_loader, train_unlabel_loader, model,
                                        optimizer, metrics, epoch, global_step, device)
        for k in train_result.keys():
            if ('train_' + k) not in records:
                records['train_' + k] = []
            records['train_' + k].append(train_result[k])
        
        if (epoch + 1) % train_cfg['VAL_STEP'] == 0:
            val_result = val_epoch(train_cfg, val_label_loader, model, metrics, epoch, device)
        else:
            val_result = {
                'loss_all': 0, 'focal_loss': 0, 'dice_loss': 0,
                'OverallAccuracy': 0, 'OverallPrecision': 0,
                'OverallRecall': 0, 'AverageAccuracy': 0,
                'AveragePrecision': 0, 'AverageRecall': 0,
                'IoU': 0, 'F1Score': 0,
            }
        for k in val_result.keys():
            if ('val_' + k) not in records:
                records['val_' + k] = []
            records['val_' + k].append(val_result[k])

        if records['val_IoU'][-1] > best_IoU:
            best_IoU = records['val_IoU'][-1]
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }
            torch.save(state, os.path.join(output_folder, 'epoch{:02d}.pth.tar'.format(epoch + 1)))

        scheduler.step()

    records = pd.DataFrame(records)
    records.to_csv(os.path.join(train_cfg['OUT_PATH'], 'records.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description='SemSeg')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help='Path of config files')
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    yaml_path = args.cfg_file

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_cfg = config['MODEL']
    dataset_cfg = config['DATASET']
    optimizer_cfg = config['OPTIMIZER']
    train_cfg = config['TRAIN']
    mertrics_cfg = config['METRICS']

    output_folder = train_cfg['OUT_PATH']
    os.makedirs(output_folder, exist_ok=True)
    os.system('cp {} {}/config.yaml'.format(yaml_path, output_folder))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train(model_cfg, optimizer_cfg, dataset_cfg, train_cfg, mertrics_cfg, device)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
