from pytorch_lightning import Trainer
from bttr.datamodule import UniDatamodule3
from bttr.lit_bttr import LitBTTR

ckp_path = 'epoch=73-step=1110000-mean_cer=0.1213.ckpt','epoch=59-step=900000-mean_cer=0.1298.ckpt'

if __name__ == "__main__":

    heights=list(range(32,129,8))
    datasets=['hme100k']

    for scaled_height in heights:
        trainer = Trainer(logger=False, accelerator='gpu')

        dm = UniDatamodule3(datasets = datasets, scaled_height=scaled_height, batch_size = 1, num_workers = 4)

        model = LitBTTR.load_from_checkpoint('lightning_logs/version_15/checkpoints/'+ckp_path)

        model.scaled_height=scaled_height
        model.ckpt_id=ckp_path.split('-step=')[0].split('=')[1]
        print('scaled_height',model.scaled_height)
        print('ckpt_id',model.ckpt_id)
        trainer.test(model, datamodule=dm)