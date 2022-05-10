import os
import json
import torch
import shutil
import argparse
from loguru import logger
import numpy as np
from torch.cuda import amp
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class WBDataset(Dataset):
    def __init__(self, path, tokenizer='bert-base-chinese'):
        self.data_list = self._read_file(path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.label = {'happy': 0, 'angry': 1, 'sad': 2, 'fear': 3, 'surprise': 4, 'neutral': 5}

    def _read_file(self, path):
        with open(path, 'r') as f:
            data_list = json.load(f)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        content = data['content']
        label = self.label[data['label']]
        token = self.tokenizer(content, padding='max_length', truncation=True, max_length=140)
        input_ids = token['input_ids']
        token_type_ids = token['token_type_ids']
        attention_mask = token['attention_mask']
        return input_ids, token_type_ids, attention_mask, label

    @staticmethod
    def collate_fn(batch):
        input_ids, token_type_ids, attention_mask, label = list(zip(*batch))
        input_ids = torch.tensor(input_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        label = torch.tensor(label)
        return input_ids, token_type_ids, attention_mask, label


class Trainer():
    def __init__(self, train_dataloader, model, lr, epochs, output_dir, val_dataloader=None, save_model_iter=5):
        self.device = torch.device('cuda:0')
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.n_batch = len(train_dataloader)
        self.model = model.to(self.device)
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.num_training_steps = self.n_batch * epochs
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_training_steps=self.num_training_steps,
                                                         num_warmup_steps=self.n_batch)
        self.scaler = amp.GradScaler(enabled=True)
        self.epochs = epochs
        self.start_epoch = 0
        self.global_step = 0
        self.save_model_iter = save_model_iter
        self.output_dir = output_dir
        self._init_logger()

        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        self.metrics = {'val_loss': 0, 'val_acc': 0, 'val_f1_score': 0}

    def _init_logger(self):
        log_file_name = 'train_{time}.log'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        log_path = os.path.join(self.output_dir, log_file_name)
        logger.add(log_path, rotation='1 week', retention='30 days', enqueue=True)

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(enumerate(self.train_dataloader), total=self.n_batch)
        train_loss = AverageMeter('Loss', ':6.3f')
        acc = AverageMeter('acc', ':6.3f')
        f1 = AverageMeter('f1_score', ':6.3f')

        for index, (input_ids, token_type_ids, attention_mask, label) in pbar:
            self.global_step += 1
            output = model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device),
                           labels=label.to(self.device))
            loss = output.loss
            logits = output.logits

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            train_f1_score = f1_score(y_true=label.cpu().numpy(), y_pred=preds, average='macro')
            train_acc = accuracy_score(y_true=label.cpu().numpy(), y_pred=preds)
            train_loss.update(loss.detach().cpu().numpy().tolist())
            acc.update(train_acc)
            f1.update(train_f1_score)
            pbar.set_description(f'{train_loss.avg: .3f}  {acc.avg: .3f}  {f1.avg: .3f}')
            pbar.set_postfix({'epoch': str(epoch) + '/' + str(self.epochs),
                              'lr': self.optimizer.state_dict()['param_groups'][0]['lr']})

            self.writer.add_scalar('Train/loss', train_loss.avg, self.global_step)
            self.writer.add_scalar('Train/acc', acc.avg, self.global_step)
            self.writer.add_scalar('Train/f1_score', f1.avg, self.global_step)
            self.writer.add_scalar('Train/lr', self.optimizer.state_dict()['param_groups'][0]['lr'],
                                   self.global_step)

        logger.info(f'{train_loss.avg: .3f}  {acc.avg: .3f}  {f1.avg: .3f}')

    def on_epoch_finish(self, epoch):
        save_best = False
        metric = None
        if self.val_dataloader is not None:
            with torch.no_grad():
                val_loss, val_acc, val_f1_score = self._val()
            self.writer.add_scalar('Val/loss', val_loss, epoch)
            self.writer.add_scalar('Val/acc', val_acc, epoch)
            self.writer.add_scalar('Val/f1_score', val_f1_score, epoch)
            metric = {'val_loss': val_loss, 'val_acc': val_acc, 'val_f1_score': val_f1_score}
            logger.info(f'complete {epoch} epochs, val result: {metric}')
            if val_f1_score > self.metrics['val_f1_score']:
                save_best = True
                self.metrics = {'val_loss': val_loss, 'val_acc': val_acc, 'val_f1_score': val_f1_score}

        if save_best:
            self._save_checkpoint(epoch, metric, save_best)

        elif epoch % self.save_model_iter == 0:
            self._save_checkpoint(epoch, metric)
        else:
            pass

    def _save_checkpoint(self, epoch, metric, save_best=False):

        state_dict = self.model.state_dict()
        state = {
            'epoch': epoch,
            'metric': metric,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        name = f'{epoch}.pt'
        path = os.path.join(self.output_dir, name)
        torch.save(state, path, _use_new_zipfile_serialization=False)
        if save_best:
            shutil.copy(path, os.path.join(self.output_dir, 'best.pt'))
            logger.info(f'Saving current best: {path}')
        else:
            logger.info(f'Saving checkpoint: {path}')

    def train(self):
        logger.info('===============================START===============================')
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            print('\n loss    acc    f1_score')
            self.train_epoch(epoch)
            self.on_epoch_finish(epoch)
        logger.info(f'complete the training, best result: {self.metrics}')
        logger.info('===============================END===============================')

    def _val(self):
        self.model.eval()
        pbar = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader))

        preds = []
        labels = []
        losses = []
        for index, (input_ids, token_type_ids, attention_mask, label) in pbar:
            output = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device),
                                labels=label.to(self.device))

            loss = output.loss
            logits = output.logits
            losses.append(loss.detach().cpu().numpy().tolist())
            preds.extend(np.argmax(logits.detach().cpu().numpy(), axis=1).tolist())
            labels.extend(label.cpu().numpy().tolist())
        val_f1_score = f1_score(y_true=np.array(labels), y_pred=np.array(preds), average='macro')
        val_acc = accuracy_score(y_true=np.array(labels), y_pred=np.array(preds))
        val_loss = np.array(losses).mean()
        return val_loss, val_acc, val_f1_score


def parse_args():
    parser = argparse.ArgumentParser(description='Bert fine-tune on SMP2020-EWECT Dataset')
    parser.add_argument('--model_name', default='bert-base-chinese', type=str,
                        help='huggingface transformer model name')
    parser.add_argument('--num_labels', default=6, type=int, help='fine-tune num labels')
    parser.add_argument('--train_data_path', default='data/clean/usual_train.txt', type=str, help='train data path')
    parser.add_argument('--val_data_path', default='data/clean/usual_eval_labeled.txt', type=str,
                        help='train data path')
    parser.add_argument('--batch_size', default=64, type=int, help='train and validation batch size')
    parser.add_argument('--dataloader_num_workors', default=8, type=int, help='pytorch dataloader num workers')
    parser.add_argument('--lr', default=1e-5, type=float, help='train learning rate')
    parser.add_argument('--epochs', default=30, type=int, help='train epochs')
    parser.add_argument('--output_dir', default='workspace/wb', type=str, help='save dir')
    parser.add_argument('--save_model_iter', default=5, type=int, help='save model num epochs on training')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    train_dataset = WBDataset(args.train_data_path, args.model_name)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_num_workors,
                                  collate_fn=WBDataset.collate_fn)

    val_dataset = WBDataset(args.val_data_path, args.model_name)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.dataloader_num_workors,
                                collate_fn=WBDataset.collate_fn, drop_last=False)

    trainer = Trainer(train_dataloader=train_dataloader, model=model, lr=args.lr, epochs=args.epochs,
                      output_dir=args.output_dir, save_model_iter=args.save_model_iter, val_dataloader=val_dataloader)
    trainer.train()
