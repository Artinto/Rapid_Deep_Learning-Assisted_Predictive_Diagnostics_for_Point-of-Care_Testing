import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from copy import deepcopy
from PIL import Image
from collections import defaultdict

from dataset import KitDataset
from models import main_model, weights_init
from utils import mk_dir, Log
from utils.roc_curve import plot_roc_curve


def main(args):
    # seed & device
    torch.manual_seed(args.seed)
    device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    log = Log(args)
    for k, v in args.__dict__.items():
        log.logging(f'{k}: {v}')
    log.logging(f'Real used device: {device}')

    data_loaders = {}
    if args.mode == 'train':
        data_loaders['train'] = DataLoader(
            dataset=KitDataset(root=os.path.join(args.data_path, 'train'), args=args, is_train=True),
            batch_size=args.train_batch_size,
            shuffle=True
        )
    data_loaders['eval'] = DataLoader(
        dataset=KitDataset(root=os.path.join(args.data_path, 'eval'), args=args, is_train=False),
        batch_size=args.eval_batch_size
    )
    log.logging('\n')
    log.print_table(header=['dataset']+list(data_loaders.keys()), table=[['num']+[len(data_loaders[k].dataset) for k in data_loaders.keys()]])
    log.logging('\n')

    # define models
    model = main_model(args)
    if args.load_saved_model:
        state = torch.load(args.path_saved_model)
        for k, v in state.copy().items():
            if 'module' in k:
                state[k.replace('module.', '')] = v
                del (state[k])
        model.load_state_dict(state)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    if not args.load_saved_model: model.apply(weights_init)

    # loss function & optimizer
    criterion = dict()
    if args.criterion == 'mse':
        criterion['reg'] = nn.MSELoss().to(device)
    elif args.criterion == 'l1':
        criterion['reg'] = nn.L1Loss().to(device)
    criterion['cls'] = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    if args.mode == 'train':
        model_train(log, args, data_loaders, model, criterion, optimizer, device)
    elif args.mode == 'test':
        model_evaluation(args, log, model, data_loaders, criterion, device)


def model_train(log, args, data_loaders, model, criterion, optimizer, device):
    best_reg_rmse, best_cls_acc = 1e+10, 0
    for epoch_idx in range(1, args.num_epochs+1):
        log.logging('\n\n'+'-'*65+f' {epoch_idx} epoch start! '+'-'*65)

        train_reg_loss, train_cls_loss, train_reg_rmse, train_cls_acc = train_one_epoch(log, args, epoch_idx, data_loaders, model, criterion, optimizer, device)
        eval_reg_loss, eval_cls_loss, eval_reg_rmse, eval_cls_acc, y_real, y_pred = evaluation(log, model, data_loaders['eval'], criterion, device)

        log.logging(
            f'\n>> epoch: {epoch_idx:2d}\n'
            f'\t* train || reg_loss : {train_reg_loss:.4f} | cls_loss : {train_cls_loss:.4f} |  rmse: {train_reg_rmse:0.4f} | target_acc: {train_cls_acc:6.2f}%\n'
            f'\t* eval  || reg_loss : { eval_reg_loss:.4f} | cls_loss : { eval_cls_loss:.4f} |  rmse: { eval_reg_rmse:0.4f} | target_acc: { eval_cls_acc:6.2f}%\n'
        )

        log.writer.add_scalars(
            'reg_loss', {'train_reg_loss': train_reg_loss,
                         'eval_reg_loss':  eval_reg_loss}, epoch_idx
        )
        log.writer.add_scalars(
            'cls_loss', {'train_cls_loss': train_cls_loss,
                         'eval_cls_loss':  eval_cls_loss}, epoch_idx
        )
        log.writer.add_scalars(
            'reg_rmse', {'train_reg_rmse': train_reg_rmse,
                         'eval_reg_rmse': eval_reg_rmse}, epoch_idx
        )
        log.writer.add_scalars(
            'target_acc', {'train_target_acc': train_cls_acc,
                           'eval_target_acc': eval_cls_acc}, epoch_idx
        )

        save_model_flag = []
        if best_reg_rmse >= eval_reg_rmse:
            best_reg_rmse = eval_reg_rmse
            log.writer.add_text('best reg', f'rmse: {eval_reg_rmse:.4f} | target acc: {eval_cls_acc:.4f}%', epoch_idx)
            save_model_flag.append('density')

        if best_cls_acc <= eval_cls_acc:
            best_cls_acc = eval_cls_acc
            log.writer.add_text('best target', f'rmse: {eval_reg_rmse:.4f} | target acc: {eval_cls_acc:.4f}%', epoch_idx)
            save_model_flag.append('target')

        if args.save_model and save_model_flag and epoch_idx > args.th_epoch:
            for flg in save_model_flag:
                torch.save(model.state_dict(), f'./{log.model_save_dir}/best_{flg}_model.pt')
                with open(f'./{log.model_save_dir}/best_{flg}_model.txt', 'w') as f:
                    f.write(f'epoch:{epoch_idx:5d} | rmse: {eval_reg_rmse:.3f} | target acc: {eval_cls_acc:.3f}%')

    if args.save_model:
        torch.save(model.state_dict(), f'./{log.model_save_dir}/final_model.pt')
        with open(f'./{log.model_save_dir}/final_model.txt', 'w') as f:
            f.write(f'epoch:{epoch_idx:5d} | rmse: {eval_reg_rmse:.3f} | target acc: {eval_cls_acc:.3f}%')
    log.writer.close()


def train_one_epoch(log, args, epoch_idx, data_loaders, model, criterion, optimizer, device):
    model.train()
    train_reg_loss = 0
    train_cls_loss = 0
    total_rss, total_cls_correct = 0, 0
    num_batches = len(data_loaders['train'])

    for batch_idx, (imgs, label_dsts, label_clss) in enumerate(data_loaders['train'], 1):
        imgs, label_dsts, label_clss = imgs.to(device), label_dsts.to(device), label_clss.to(device)
        reg_preds = model(imgs)

        # Regression & Classification loss
        reg_loss = criterion['reg'](reg_preds, label_dsts)
        cls_loss = 0
        loss = reg_loss # + cls_loss
        loss.backward()

        train_reg_loss += reg_loss / num_batches

        reg_rss, label_rss = calc_reg_rmse(reg_preds, label_dsts)
        total_rss += reg_rss
        batch_rmse = (reg_rss/len(label_clss))**0.5

        cls_correct, cls_acc, cls_pred, cls_real = calc_cls_acc(reg_preds, label_clss)
        total_cls_correct += cls_correct

        optimizer.step()
        optimizer.zero_grad()

        log.logging(
            f'Epoch: {epoch_idx:3d}/{args.num_epochs}  |  Batch: {batch_idx:3d}/{num_batches}  |  '
            f'batch_reg_loss: {reg_loss:.3f}  |  batch_cls_loss: {cls_loss:.3f}  |  '
            f'batch_reg_rmse: {batch_rmse:.3f}  |  batch_target_acc: {cls_acc:6.2f}%'
        )

    train_reg_rmse = (total_rss / len(data_loaders['train'].dataset))**0.5
    train_cls_acc = total_cls_correct / len(data_loaders['train'].dataset) * 100
    return train_reg_loss, train_cls_loss, train_reg_rmse, train_cls_acc


def model_evaluation(args, log, model, data_loaders, criterion, device):
    eval_reg_loss, eval_cls_loss, eval_reg_rmse, eval_cls_acc, y_real, y_pred = evaluation(log, model, data_loaders['eval'], criterion, device)
    log.logging('\n')
    log.logging(f'eval_reg_loss: {eval_reg_loss:.4f}\t|\t eval_cls_loss: {eval_cls_loss:.4f}\t|'
                f'\t eval_reg_rmse: {eval_reg_rmse:.4f}\t|\t eval_target_acc: {eval_cls_acc:.2f}%')
    log.logging('\n')
    if args.save_roc_curve:
        plot_roc_curve(y=y_real, y_pred=y_pred, save_path=log.log_dir)


def evaluation(log, model, eval_loader, criterion, device):
    model.eval()
    total_reg_rss, total_cls_c = 0, 0
    total_reg_rmse_dict = defaultdict(float)
    reg_rss_dict = dict()
    eval_reg_loss, eval_cls_loss = 0, 0
    y_real, y_pred = np.empty([0]), np.empty([0])

    with torch.no_grad():
        kit_id = 0
        f = open(log.log_dir + 'reg_preds_and_label.txt', 'w')
        for batch_idx, (imgs, label_dsts, label_clss) in enumerate(eval_loader, 1):
            imgs, label_dsts, label_clss = imgs.to(device), label_dsts.to(device), label_clss.to(device)
            reg_preds = model(imgs)

            reg_loss = criterion['reg'](reg_preds, label_dsts)
            eval_reg_loss += reg_loss/len(eval_loader)

            cls_loss = 0 # criterion['cls'](cls_preds, label_clss)
            eval_cls_loss += cls_loss/len(eval_loader)

            reg_rss, label_rss = calc_reg_rmse(reg_preds, label_dsts)
            total_reg_rss += reg_rss

            for k, v in label_rss.items():
                if k in reg_rss_dict:
                    sum_rss = reg_rss_dict[k][0] + v[0]
                    sum_counts = reg_rss_dict[k][1] + v[1]
                else:
                    sum_rss = v[0]
                    sum_counts = v[1]
                reg_rss_dict[k] = (sum_rss, sum_counts)

            cls_correct, cls_acc, cls_pred, cls_real = calc_cls_acc(reg_preds, label_clss)
            total_cls_c += cls_correct

            y_pred = np.concatenate([y_pred, cls_pred.cpu()])
            y_real = np.concatenate([y_real, cls_real.cpu()])

            # 예측값 정답 파일 저장

            kit_num =eval_loader.dataset.all_data[kit_id]
            kit_id += 1
            r_p = reg_preds.cpu().numpy()
            l_d = label_dsts.cpu().numpy()
            f.write(str(kit_num['img_files'][0].split('/')[-2]) + ' ')
            f.write(str(l_d[0]) + ' ')
            f.write(str(r_p[0]) + '\n')
        f.close()


    for label, (se, counts) in reg_rss_dict.items():
        total_reg_rmse_dict[label] = (se/counts)**0.5

    eval_reg_rmse = (total_reg_rss / len(eval_loader.dataset))**0.5
    eval_cls_acc = total_cls_c / len(eval_loader.dataset) * 100
    log.print_label_rmse_table(total_reg_rmse_dict)
    return eval_reg_loss, eval_cls_loss, eval_reg_rmse, eval_cls_acc, y_real, y_pred


def calc_reg_rmse(pred, labels):
    preds, labels = pred.data.cpu().numpy(), labels.data.cpu().numpy()
    # square_error
    se = ((preds - labels)**2).sum()
    label_rss = defaultdict(tuple)
    unique, counts = np.unique(labels, return_counts=True)
    for k, v in zip(unique, counts):
        indices = np.where(labels==k)[0]
        preds_k = np.take(preds, indices)
        labels_k = np.take(labels, indices)
        label_rss[k] = (((preds_k - labels_k)**2).sum(), v)
    return se, label_rss


def calc_cls_acc(pred, labels):
    corr = (pred > 3.5).float() == labels
    cls_correct = corr.float().sum()
    cls_acc = corr.float().mean()*100
    return cls_correct, cls_acc, (pred > 3.5).float(), labels
