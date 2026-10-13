"""Train/evaluate EEG and export paired-subject 256-D features."""
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from emmnet_eeg import EEGClassifier
from eeg_data import load_splits, EMMNetEEGDataset, fit_statistics
from eeg_loss import FocalLoss

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def forward_crops(model, batch, device, features=False):
    x, age = batch['eeg'].to(device), batch['age'].to(device)
    if x.ndim == 3:
        return model.encoder(x,age) if features else model(x,age)
    b,k,c,t = x.shape
    fn = model.encoder if features else model
    out = fn(x.reshape(b*k,c,t), age.repeat_interleave(k)).reshape(b,k,-1)
    # Subject predictions average probabilities, not class indices or logits.
    return out.mean(1) if features else out.softmax(-1).mean(1)

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    confusion = torch.zeros(num_classes,num_classes,dtype=torch.long)
    predictions = []
    for batch in loader:
        probs = forward_crops(model,batch,device)
        pred = probs.argmax(1).cpu()
        for sid,y,p,score in zip(batch['subject_id'],batch['label'],pred,probs.cpu()):
            confusion[y,p] += 1
            predictions.append({'subject_id':sid,'label':int(y),'prediction':int(p),'probabilities':score.tolist()})
    rows = confusion.sum(1)
    sensitivity = [float(confusion[i,i]/rows[i]) if rows[i] else None for i in range(num_classes)]
    return {'accuracy':float(confusion.diag().sum()/confusion.sum()),
            'class_sensitivity':sensitivity,'confusion_matrix':confusion.tolist(),'subjects':predictions}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','evaluate','features'])
    parser.add_argument('--split-json',required=True)
    parser.add_argument('--annotation',help='annotation.json supplying ages absent from split JSON')
    parser.add_argument('--eeg-dir',required=True)
    parser.add_argument('--output',default='local/eeg-run')
    parser.add_argument('--checkpoint')
    parser.add_argument('--num-classes',type=int,choices=[2,3],default=2)
    parser.add_argument('--include-ecg',action='store_true')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--weight-decay',type=float,default=0.01)
    parser.add_argument('--warmup-ratio',type=float,default=0.05)
    parser.add_argument('--noise-std',type=float,default=0.001)
    parser.add_argument('--eval-crops',type=int,default=8)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--split',choices=['train','val','test'],default='test')
    args = parser.parse_args()
    if args.batch_size < 2 or args.epochs < 1 or args.lr <= 0 or not 0 <= args.warmup_ratio < 1:
        parser.error('batch-size >= 2, epochs >= 1, lr > 0, 0 <= warmup-ratio < 1 required')
    if args.mode != 'train' and not args.checkpoint:
        parser.error('--checkpoint required for evaluate/features')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    ckpt = torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    if ckpt:
        if args.mode == 'train':
            parser.error('Training resume is not supported; use evaluate/features for checkpoints')
        args.num_classes = ckpt['model_options']['num_classes']
        args.include_ecg = ckpt['model_options']['eeg_channels'] == 20
        args.eval_crops = ckpt['options']['eval_crops']
    splits = load_splits(args.split_json,args.annotation,args.num_classes)
    datasets = {s:EMMNetEEGDataset(items,args.eeg_dir,s,include_ecg=args.include_ecg,
                                  noise_std=args.noise_std,eval_crops=args.eval_crops)
                for s,items in splits.items()}
    stats = ckpt['statistics'] if ckpt else fit_statistics(datasets['train'])
    for dataset in datasets.values():
        dataset.stats = stats
        if args.mode != 'train':
            dataset.split = 'test'
    options = {'num_classes':args.num_classes,'eeg_channels':20 if args.include_ecg else 19}
    model = EEGClassifier(**options).to(args.device)
    if ckpt:
        model.load_state_dict(ckpt['model_state'])
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s:DataLoader(d,batch_size=args.batch_size,shuffle=s=='train' and args.mode=='train',
                           drop_last=s=='train' and args.mode=='train',num_workers=args.workers,
                           worker_init_fn=seed_worker,generator=generator) for s,d in datasets.items()}
    if args.mode == 'train':
        if len(loaders['train']) == 0:
            parser.error('Training subjects must be at least batch-size (BatchNorm/drop_last)')
        criterion = FocalLoss().to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,betas=(0.9,0.999))
        total = args.epochs*len(loaders['train'])
        warmup = int(total*args.warmup_ratio)
        def lr_factor(step):
            if warmup and step < warmup:
                return (step+1)/warmup
            progress = (step-warmup)/max(1,total-warmup-1)
            return 0.5*(1+math.cos(math.pi*min(1,progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
        history, best = [], -1
        for epoch in range(args.epochs):
            model.train()
            loss_sum = 0
            for batch in loaders['train']:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(forward_crops(model,batch,args.device),batch['label'].to(args.device))
                loss.backward(); optimizer.step(); scheduler.step()
                loss_sum += float(loss.detach())
            metrics = evaluate(model,loaders['val'],args.device,args.num_classes)
            entry = {'epoch':epoch+1,'train_loss':loss_sum/len(loaders['train']),
                     'validation_accuracy':metrics['accuracy']}
            history.append(entry)
            print(json.dumps(entry),flush=True)
            if metrics['accuracy'] > best:
                best = metrics['accuracy']
                torch.save({'model_state':model.state_dict(),'model_options':options,'statistics':stats,
                            'options':vars(args),'epoch':epoch+1,'validation_accuracy':best,
                            'subject_splits':{s:[x['serial'] for x in v] for s,v in splits.items()}},output/'best.pt')
            (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        ckpt = torch.load(output/'best.pt',map_location=args.device,weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        (output/'test_metrics.json').write_text(json.dumps(evaluate(model,loaders['test'],args.device,args.num_classes),indent=2),encoding='utf-8')
    elif args.mode == 'evaluate':
        # Reject evaluation leakage against the checkpoint's training subjects.
        overlap = set(ckpt['subject_splits']['train']) & {x['serial'] for x in splits[args.split]}
        if args.split != 'train' and overlap:
            raise ValueError(f'Evaluation subjects were in checkpoint training: {sorted(overlap)}')
        metrics = evaluate(model,loaders[args.split],args.device,args.num_classes)
        (output/f'{args.split}_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print('accuracy:',metrics['accuracy'])
    else:
        # Deterministic multicrop feature extraction, including the train split.
        datasets[args.split].split = 'test'
        model.eval()
        vectors, ids, labels = [], [], []
        with torch.no_grad():
            for batch in loaders[args.split]:
                vectors.append(forward_crops(model,batch,args.device,features=True).cpu().numpy())
                ids.extend(batch['subject_id']); labels.extend(batch['label'].tolist())
        np.savez_compressed(output/f'{args.split}_features.npz',features=np.concatenate(vectors),
                            subject_ids=np.asarray(ids),labels=np.asarray(labels))

if __name__ == '__main__':
    main()
