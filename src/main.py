import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
ckp = utility.Checkpoint(args)


def main():
    if ckp.ok:
        loader = data.Data(args)
        _model = model.Model(args, ckp)
        # from thop import profile
        # flops, params = profile(_model, inputs=(torch.randn(1, 3, 128, 128).cuda(), 0))
        # print(flops)
        # print(params)
        # NLAN  407520706560.0
        # ENLAN 363017830400.0

        print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters()) / 1000000.0))
        _loss = loss.Loss(args, ckp) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, ckp)
        while not t.terminate():
            t.train()
            t.test()


        ckp.done()

# from torchstat import stat
# from torchvision.models.resnet import resnet34
# model = resnet34()
# stat(model, (3, 224, 224))


# # Example X2 SR
# !python main.py --dir_data ../ --n_GPUs 1 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4
# --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1
# --batch_size 16 --model NLSN --scale 2 --patch_size 96 --save_dir /content/drive/MyDrive/sr_output/NLSN
# --save NLSN_x2 --data_train DIV2K --load NLSN_x2
if __name__ == '__main__':
    main()
