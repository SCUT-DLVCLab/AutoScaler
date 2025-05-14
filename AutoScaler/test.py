from pytorch_lightning import Trainer
from bttr.datamodule import UniDatamodule3
from bttr.lit_bttr import LitBTTR
import os


ckp_path = "lightning_logs/version_0/checkpoints/epoch=11-step=360000-val_loss=0.2322.ckpt"

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # datasets=['hme100k','mlhme38k','csdb_mini','zinc','table_bank','didi','didi_no_text','primus']
    datasets=['hme100k']
    trainer = Trainer(logger=False, accelerator='gpu')

    # dm = UniDatamodule3(datasets = datasets, batch_size = 1, num_workers = 64, scaled_heights=list(range(192,337,8)))

    dm = UniDatamodule3(datasets = datasets, batch_size = 1, num_workers = 0, scaled_heights=[80])

    model = LitBTTR.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)