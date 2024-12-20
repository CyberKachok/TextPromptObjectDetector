import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from lib.utils.prior_boxes import detect_objects, prior_boxes
from lib.utils.multibox_loss import MultiBoxLoss
from open_clip import tokenizer
import schedulefree


def save_model_if_best(model, best_loss, eval_loss, args, custom_config):
    if eval_loss < best_loss:
        torch.save(model.state_dict(), os.path.join(args.output, f"{custom_config['model_name']}.pth"))
        return eval_loss
    return best_loss


def process_batch(model, data_loader, prior_box_s_gpu, criterion, optimizer, train=True):
    total_loc_loss, total_cls_loss, total_loss = 0, 0, 0
    if train:
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    scaler = torch.cuda.amp.GradScaler()
    bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (image_s_cpu, box_ss_cpu, label_ss_cpu) in bar:
        # for k in range(1, 3):
        input = prepare_input(image_s_cpu, label_ss_cpu, data_loader.dataset.dataset.caption_i2l)
        for l in label_ss_cpu:
            l[l != 1] = 1

        # Move data to GPU
        label_ss_gpu = [label_s_cpu.cuda() for label_s_cpu in label_ss_cpu]
        box_ss_gpu = [box_ss_cpu.cuda() for box_ss_cpu in box_ss_cpu]

        if all(len(l) > 0 and len(b) > 0 for l, b in zip(label_ss_gpu, box_ss_gpu)):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_loc_ss_gpu, pred_conf_ss_gpu = model(input)
            loc_loss, cls_loss = criterion((pred_loc_ss_gpu, pred_conf_ss_gpu, prior_box_s_gpu),
                                           (label_ss_gpu, box_ss_gpu))
            loss = loc_loss + cls_loss

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            total_loc_loss += loc_loss.item()
            total_cls_loss += cls_loss.item()
            total_loss += loss.item()
            running_loss += loss.item()
        if (step + 1) % 10 == 0:
            bar.set_postfix(Train_Loss=torch.tensor(running_loss / (step + 1)), LR=optimizer.param_groups[0]['lr'])
    # break
    return total_loc_loss, total_cls_loss, total_loss


def evaluate_model(model, dataloader, map_metric, prior_box_s, custom_config):
    model.eval()
    map_metric.reset()
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for step, (image_s_cpu, box_ss_cpu, label_ss_cpu) in bar:
            input = prepare_input(image_s_cpu, label_ss_cpu, dataloader.dataset.dataset.caption_i2l)
            for l in label_ss_cpu:
                l[l != 1] = 1

            label_ss_gpu = [label_s_cpu.cuda() for label_s_cpu in label_ss_cpu]
            box_ss_gpu = [box_ss_cpu.cuda() for box_ss_cpu in box_ss_cpu]

            if all(len(l) > 0 and len(b) > 0 for l, b in zip(label_ss_gpu, box_ss_gpu)):
                pred_locs, pred_confs = model(input)

                pred_confs = F.softmax(pred_confs, dim=2).cpu()
                pred_locs = pred_locs.cpu()
                pred_boxes, pred_labels, pred_scores = detect_objects(
                    pred_locs, pred_confs, prior_box_s,
                    custom_config['num_classes'], custom_config['overlap_threshold'], 0.25)

                preds = [{'boxes': boxes.cpu(), 'labels': labels.cpu(), 'scores': scores.cpu()}
                         for boxes, labels, scores in zip(pred_boxes, pred_labels, pred_scores)]
                labels = [{'boxes': boxes, 'labels': labels}
                          for boxes, labels in zip(box_ss_cpu, label_ss_cpu)]
                map_metric.update(preds, labels)
        # if (step + 1) % 100 == 0:
        # 	bar.set_postfix(step=step)

        return map_metric.compute()["map"]


def prepare_input(image_s_cpu, k, dct):
    # print(k)
    # print([dct[el[-1].item()] for el in k])
    # print(tokenizer.tokenize([dct[el[-1].item()] for el in k], context_length=225))
    return {
        'template': tokenizer.tokenize([dct[el[-1].item()] for el in k], context_length=77),
        'search': image_s_cpu,
        'label_cls': None,
        'label_loc': None,
        'label_loc_weight': None,
        'bbox': None
    }


def train_process(model, dataloaders, args, custom_config, logger=None, experiment_name=''):
    if logger is not None:
        run_id = logger.start_run(experiment_name)

    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    train_dataloader = dataloaders['train']
    test_dataloader = dataloaders['test']

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    prior_box_s = prior_boxes(custom_config).cuda()
    criterion = MultiBoxLoss(custom_config['overlap_threshold'], custom_config['neg_pos_ratio'],
                             custom_config['variance'])

    # optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.learning_rate * 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate * 1, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.2)
    map_metric = MeanAveragePrecision(iou_thresholds=[0.5])

    model.cuda()
    criterion.cuda()

    train_loss_s, eval_loss_s = [], []
    best_loss = np.inf
    # test_dataloader = train_dataloader
    # train_dataloader = test_dataloader
    for epoch in range(args.epochs):
        # optimizer.train()
        train_loc_loss, train_cls_loss, train_loss = process_batch(model, train_dataloader, prior_box_s, criterion,
                                                                   optimizer, train=True)
        # optimizer.eval()
        with torch.no_grad():
            eval_loc_loss, eval_cls_loss, eval_loss = process_batch(model, test_dataloader, prior_box_s, criterion,
                                                                    optimizer, train=False)

        train_loss /= len(train_dataloader)
        eval_loss /= len(test_dataloader)

        eval_cls_loss /= len(test_dataloader)
        eval_loc_loss /= len(test_dataloader)

        train_cls_loss /= len(train_dataloader)
        train_loc_loss /= len(train_dataloader)

        train_loss_s.append(train_loss)
        eval_loss_s.append(eval_loss)

        # scheduler.step()

        mAP = evaluate_model(model, test_dataloader, map_metric, prior_box_s.cpu(), custom_config)
        print(
            f'epoch[{epoch}] | mAP: {mAP:.4f} | loc_loss [{train_loc_loss:.2f}/{eval_loc_loss:.2f}] | cls_loss [{train_cls_loss:.2f}/{eval_cls_loss:.2f}] | total_loss [{train_loss:.2f}/{eval_loss:.2f}]')

        best_loss = save_model_if_best(model, best_loss, eval_loss, args, custom_config)

    return model, prior_box_s.cpu(), train_loss_s, eval_loss_s
