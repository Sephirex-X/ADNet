import time
import torch
from tqdm import tqdm
import numpy as np
import random
import mmcv
from adnet.models.registry import build_net
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from adnet.datasets import build_dataloader
from adnet.utils.recorder import build_recorder
from adnet.utils.net_utils import save_model, load_network
from mmcv.parallel import MMDataParallel 

def setup_seed(seed):
    print('seed: ',seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Runner(object):
    def __init__(self, cfg):
        # torch.manual_seed(cfg.seed)
        # np.random.seed(cfg.seed)
        # random.seed(cfg.seed)
        setup_seed(cfg.seed)
        self.cfg = cfg
        self.metric = 0.
        self.val_loader = None
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.device = "cuda"
        self.net = MMDataParallel(
                self.net, device_ids = range(self.cfg.gpus)).to(self.device)
        # self.net = self.net.to(self.device)
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.warmup_scheduler = None
        self.resume()
       
    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net,self.optimizer,self.scheduler,self.recorder, self.cfg.load_from,
                finetune_from=self.cfg.finetune_from, logger=self.recorder.logger,cfg=self.cfg)
        self.recorder.logger.info('Resume from: epoch' + str(self.recorder.epoch))
        # resume should update metric
        # self.validate()
    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].to(self.device)
        return batch
    
    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(mmcv.track_iter_progress(train_loader)):
            # if self.recorder.step >= self.cfg.total_iter:
            #     break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            # with self.warmup_scheduler.dampening():
            #     self.scheduler.step()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)
            self.recorder.write_tensorboard(output['loss_stats'],scalar=epoch*max_iter + i)
            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')
                self.recorder.write_tensorboard(dict(lr=lr),scalar=epoch*max_iter + i)
    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)

        self.recorder.logger.info('Start training...')
        for epoch in range(self.recorder.epoch,self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            #======== dynamic eval =========
            if epoch+1 >= self.cfg.dynamic_after:
                eval_ep = 1
            else:
                eval_ep = self.cfg.eval_ep 
            if (epoch + 1) % eval_ep == 0:
                metric = self.validate(test=False)
                if metric > self.metric:
                    self.metric = metric
                    self.save_ckpt(is_best=True)
                self.recorder.logger.info('Best metric: ' + str(self.metric))
                self.recorder.write_tensorboard(dict(val_metric=metric),scalar=epoch)
            #===== END OF SECTION ======
            if epoch == self.cfg.epochs - 1:
                metric = self.validate(test=True)
                self.recorder.logger.info('Test metric: ' + str(metric))
            # if self.recorder.step >= self.cfg.total_iter:
            #     break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()

    def validate(self,test=True):
        self.net.eval()
        if test:
            self.val_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)
        else:
            self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)
        anks = []
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output_pred = self.net.module.get_lanes(output)
                
                predictions.extend(output_pred)
            if self.cfg.view:
                out_ank = self.net.module.heads.get_lanes_temp(output)
                self.val_loader.dataset.view((output_pred,out_ank), data['meta'])
        out = self.val_loader.dataset.evaluate(predictions, self.recorder.work_dir)
        self.recorder.logger.info(out)
        metric = out
        return metric

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler,
                self.recorder, is_best)




