import torch
import torch.optim as optim
import time
import os, sys


def train(train_loader, model, criterion, optimizer, epoch, cfg, scaler=None, scheduler=None, tracker=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(cfg.DEVICE)
        target = target.to(cfg.DEVICE)

        optimizer.zero_grad()

        # Mixed precision training
        use_amp = scaler is not None and cfg.USE_AMP
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(input)
                loss = criterion(output, target)

                # Add KL divergence for variational inference
                if getattr(cfg, 'compute_kl_loss', False):
                    from utils.bayesian_layers import compute_kl_loss
                    kl_loss = compute_kl_loss(model)
                    loss = loss + cfg.KL_WEIGHT * kl_loss

            # Measure accuracy (outside autocast for stability)
            with torch.no_grad():
                acc1, acc5 = accuracy(output.float(), target, topk=(1, 5))

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            output = model(input)
            loss = criterion(output, target)

            # Add KL divergence for variational inference
            if getattr(cfg, 'compute_kl_loss', False):
                from utils.bayesian_layers import compute_kl_loss
                kl_loss = compute_kl_loss(model)
                loss = loss + cfg.KL_WEIGHT * kl_loss

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # compute gradient and do optimizer step
            loss.backward()
            optimizer.step()

        # Update metrics
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # Step OneCycleLR scheduler per batch
        if scheduler is not None and cfg.SCHEDULER.lower() == 'onecycle':
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0:
            progress.print(i)
            if cfg.VISDOM:
                cfg.vis.line(X=torch.Tensor([epoch+(i/len(train_loader))]).unsqueeze(0).cpu(),
                             Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                             env='torch',win=cfg.loss_window,name='train_loss',update='append')

        if i % cfg.SAVE_FREQ == 0:
            torch.save(model.state_dict(), './output/model/%s_%03d_%02d.cpt' % (cfg.DATASET_NAME, epoch, int(i)/1000))

    # Log epoch metrics to tracker
    if tracker:
        tracker.log_metrics({
            'loss': losses.avg,
            'acc': top1.avg,
            'acc_top5': top5.avg
        }, epoch, prefix='train/')

    return {'loss': losses.avg, 'acc': top1.avg, 'acc_top5': top5.avg}

def validate(val_loader, model, criterion, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            
            input = input.to(cfg.DEVICE)
            target = target.to(cfg.DEVICE)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.PRINT_FREQ == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    return losses.avg, top1.avg

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

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def prepare(cfg, use_arg_parser=True):
    if not os.path.isdir("./output/"):
        os.mkdir("./output/")
    if not os.path.isdir("./output/model"):
        os.mkdir("./output/model")
    if not os.path.isdir("./output/graph"):
        os.mkdir("./output/graph")
    if not cfg.TEST_MODE:
        if cfg.VISDOM:
            cfg.loss_window = cfg.vis.line(
                        Y=torch.zeros((1)).cpu(),
                        X=torch.zeros((1)).cpu(),env='torch',
                        opts=dict(xlabel='epoch',ylabel='Loss',
                                    title=cfg.DATASET_NAME+"_"
                                    +time.strftime("%m/%d %H:%M", time.localtime()),
                        legend=['train_loss','val_loss']))