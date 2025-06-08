from pytorch_lightning import Trainer
from autos.datamodule import UniDatamodule3
from autos.lit_autos import LitAutos
import os


ckp_path = "lightning_logs/version_0/checkpoints/epoch=199.ckpt"

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    datasets=['m2e']
    trainer = Trainer(logger=False, accelerator='gpu')

    dm = UniDatamodule3(datasets = datasets, batch_size = 1, num_workers = 64, scaled_heights=list(range(192,337,8)))

    model = LitAutos.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)