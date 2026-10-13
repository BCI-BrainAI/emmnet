import json
import numpy as np
import pytest
import torch
from eeg_data import EEG_CHANNELS, preprocess_signal, load_splits, EMMNetEEGDataset, fit_statistics
from emmnet_eeg import EEGClassifier, EEGEncoder, EMMNetFeatureFusion
from eeg_loss import FocalLoss

torch.set_num_threads(2)

def test_filter_resample_and_channel_order():
    fs = 400
    t = np.arange(fs*20)/fs
    signal = np.sin(2*np.pi*10*t)+np.sin(2*np.pi*100*t)
    x = np.tile(signal,(19,1))*1e-6
    out = preprocess_signal(x[::-1],fs,EEG_CHANNELS[::-1])
    assert out.shape == (19,4000)
    middle = out[0,500:-500]
    spectrum = abs(np.fft.rfft(middle))
    freq = np.fft.rfftfreq(len(middle),1/200)
    assert spectrum[np.argmin(abs(freq-10))] > 50*spectrum[np.argmin(abs(freq-100))]
    np.testing.assert_allclose(out[0],out[-1])
    with pytest.raises(ValueError,match='channel'):
        preprocess_signal(x[:-1],fs,EEG_CHANNELS[:-1])

def fixture_metadata(tmp_path):
    data = {}
    for split,items in [('train_split',[(1,30),(2,70)]),('validation_split',[(3,95)]),('test_split',[(4,100)])]:
        data[split] = []
        for sid,age in items:
            filename=f'{sid}.npz'
            t=np.arange(2400)/200
            np.savez(tmp_path/filename,signal=np.tile(np.sin(2*np.pi*10*t),(19,1))*1e-6,
                     sfreq=200,channel_names=np.array(EEG_CHANNELS))
            data[split].append({'serial':sid,'age':age,'class_label':sid%2,'eeg_file':filename})
    path=tmp_path/'split.json'
    path.write_text(json.dumps(data))
    return path,data

def test_dataset_stats_crops_and_leakage(tmp_path):
    path,data=fixture_metadata(tmp_path)
    splits=load_splits(path)
    train=EMMNetEEGDataset(splits['train'],tmp_path,noise_std=0)
    stats=fit_statistics(train)
    assert stats['age_mean']==50 and stats['age_std']==20
    train.stats=stats
    assert train[0]['eeg'].shape==(19,2000)
    val=EMMNetEEGDataset(splits['val'],tmp_path,'val',stats=stats,eval_crops=3)
    torch.testing.assert_close(val[0]['eeg'],val[0]['eeg'])
    assert val[0]['eeg'].shape==(3,19,2000)
    assert val[0]['age'].item()==2.25
    data['test_split'][0]['serial']=1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='leakage'):
        load_splits(path)

def test_exact_length_and_short_recording(tmp_path):
    path,_=fixture_metadata(tmp_path)
    splits=load_splits(path)
    dataset=EMMNetEEGDataset(splits['train'],tmp_path,seq_length=2400)
    assert dataset[0]['eeg'].shape==(19,2400)
    dataset.seq_length=2401
    with pytest.raises(ValueError,match='shorter'):
        dataset[0]

def test_focal_reference():
    logits=torch.tensor([[1.,2.],[3.,-1.]],requires_grad=True)
    labels=torch.tensor([0,1])
    p=logits.softmax(1)[torch.arange(2),labels]
    expected=(-.25*(1-p)**2*p.log()).mean()
    loss=FocalLoss()(logits,labels)
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

def test_model_age_gradient_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    model=EEGClassifier()
    x=torch.randn(2,19,2000)
    age=torch.tensor([-.5,.5],requires_grad=True)
    logits=model(x,age)
    assert logits.shape==(2,2)
    FocalLoss()(logits,torch.tensor([0,1])).backward()
    assert age.grad is not None and age.grad.abs().sum()>0
    assert model.encoder.backbone.input_stage[0].in_channels==20
    model.eval()
    with torch.no_grad():
        feature=model.encoder(x,age)
        assert feature.shape==(2,256)
        checkpoint=tmp_path/'weights.pt'
        torch.save(model.state_dict(),checkpoint)
        restored=EEGClassifier().eval()
        restored.load_state_dict(torch.load(checkpoint,weights_only=True))
        torch.testing.assert_close(model(x,age),restored(x,age))
        fusion=EMMNetFeatureFusion(eeg_channels=19).eval()
        assert fusion(x,age,feature).shape==(2,2)

def test_backbone_source_equivalence():
    from ceednet_original.resnet_1d import ResNet1D
    encoder=EEGEncoder().eval()
    original=ResNet1D(block='basic',conv_layers=[2,2,2,2],in_channels=19,out_dims=256,
                     seq_length=2000,base_channels=64,use_age='conv',fc_stages=3,
                     dropout=.1,activation='mish').eval()
    original.load_state_dict(encoder.backbone.state_dict())
    x=torch.randn(2,19,2000); age=torch.tensor([-.5,.5])
    with torch.no_grad():
        torch.testing.assert_close(encoder(x,age),original(x,age))

def test_edf_reader_contract(tmp_path,monkeypatch):
    import mne
    path,data=fixture_metadata(tmp_path)
    splits=load_splits(path)
    (tmp_path/'00001.edf').touch()
    splits['train'][0].pop('eeg_file')
    calls=[]
    class Raw:
        ch_names=list(EEG_CHANNELS)
        info={'sfreq':400.0}
        def get_data(self):
            t=np.arange(4800)/400
            return np.tile(np.sin(2*np.pi*10*t),(19,1))*1e-6
        def close(self):
            calls.append('close')
    monkeypatch.setattr(mne.io,'read_raw_edf',lambda *a,**kw: Raw())
    dataset=EMMNetEEGDataset(splits['train'],tmp_path)
    assert dataset.load_recording(0).shape==(19,2400)
    assert calls==['close']

def test_missing_age_and_annotation_join(tmp_path):
    path,data=fixture_metadata(tmp_path)
    data['train_split'][0].pop('age')
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='age'):
        load_splits(path)
    annotation=tmp_path/'annotation.json'
    annotation.write_text(json.dumps({'data':[{'serial':'00001','age':30}]}))
    assert load_splits(path,annotation)['train'][0]['age']==30
