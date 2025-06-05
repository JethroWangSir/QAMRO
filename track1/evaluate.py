import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import laion_clap
from mos_track1 import MosPredictor
from utils import get_texts_from_filename

# ✅ 修改 Dataset：不再讀 label，只讀檔名
class MyDataset(Dataset):
    def __init__(self, wavdir, list_file):
        self.wavdir = wavdir
        with open(list_file, 'r') as f:
            self.wavnames = sorted([line.strip().split(',')[0] for line in f])

    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname + '.wav')
        wav, _ = torchaudio.load(wavpath)
        if wav.size(1) > 480000:  # 16kHz * 30s
            wav = wav[:, :480000]
        return wav, wavname

    def __len__(self):
        return len(self.wavnames)

    def collate_fn(self, batch):
        wavs, wavnames = zip(*batch)
        max_len = max(wavs, key=lambda x: x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            padded = torch.nn.functional.pad(wav, (0, max_len - wav.shape[1]), 'constant', 0)
            output_wavs.append(padded)
        return torch.stack(output_wavs), wavnames

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default="/share/nas169/wago/AudioMOS/data/track1/audiomos2025-track1-eval-phase")
    parser.add_argument('--ckptdir', type=str, default='../ckpt/full/last_ckpt')
    parser.add_argument('--expname', type=str, default='../evaluation/full')
    args = parser.parse_args()

    UPSTREAM_MODEL = 'CLAP-music'
    DATADIR = args.datadir
    finetuned_checkpoint = args.ckptdir

    os.makedirs(args.expname, exist_ok=True)
    outfile = os.path.join(args.expname, 'answer.txt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if UPSTREAM_MODEL == 'CLAP-music':
        SSL_OUT_DIM = 512 
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device, amodel='HTSAT-base')
        net = MosPredictor(model, SSL_OUT_DIM).to(device)
        net.eval()
    else:
        print('*** ERROR *** ' + UPSTREAM_MODEL + ' not supported.')
        exit()

    ckpt = torch.load(finetuned_checkpoint, map_location=device)
    net.load_state_dict(ckpt)

    wavdir = os.path.join(DATADIR, 'DATA/wav')
    test_list = os.path.join(DATADIR, 'DATA/sets/eval_list.txt')

    test_set = MyDataset(wavdir, test_list)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=test_set.collate_fn)

    predictions_overall = {}
    predictions_textual = {}

    print('Starting evaluation')
    for wav, filenames in tqdm(test_loader, ncols=100):
        wav = wav.squeeze(1).to(device)
        text = get_texts_from_filename(filenames)
        with torch.no_grad():
            output1, output2 = net(wav, text)
        output1 = output1.cpu().numpy()[0][0]
        output2 = output2.cpu().numpy()[0][0]
        predictions_overall[filenames[0]] = output1
        predictions_textual[filenames[0]] = output2

    # ✅ 輸出結果
    with open(outfile, 'w') as f:
        for fname in sorted(predictions_overall.keys()):
            outl = fname.replace('.wav', '') + ',' + str(predictions_overall[fname]) + ',' + str(predictions_textual[fname]) + '\n'
            f.write(outl)

if __name__ == '__main__':
    main()
