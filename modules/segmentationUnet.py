import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from Losses.losses import DiceLoss
# torch.set_printoptions(profile="full")
THRESHOLD = 0.5


class Segmentation(pl.LightningModule):
    def __init__(self, hparams, model, inner_model):
        super(Segmentation, self).__init__()
        self.hparams = hparams
        self.UnetModel = model.to(hparams.device)
        self.accuracy = Accuracy()
        self.criterion = DiceLoss()#hparams.device)
        self.current_train_fold = hparams.current_train_fold
        print('UnetModel On cuda?: ', next(self.UnetModel.parameters()).is_cuda)


    def normalize(self, img):
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

    def create_return_dict(self, type, loss, input, file_name, gt_mask, z_hat_pred, current_fold):
        dict = {f"{type}_images_unet": (input.detach(), input.detach())}
        dict['file_name'] = file_name
        dict[f'{type}_images_pred'] = (gt_mask.detach(), z_hat_pred.detach())
        dict[f'{type}_loss'] = loss.detach()
        dict['epoch'] = self.current_epoch
        dict[f'current_{type}_fold'] = current_fold
        # dict['val_acc'] = accuracy

        return dict

    def step(self, input, gt_mask):
        input_n = self.normalize(input)
        z_hat = self.UnetModel(input_n)
        loss, z_hat_pred = self.criterion(z_hat, gt_mask)
        # z_hat_pred = torch.sigmoid(z_hat)
        accuracy = self.accuracy(z_hat_pred, gt_mask.int())

        return loss, z_hat_pred, accuracy

    def training_step(self, batch, batch_idx):
        print('TRAINING STEP ')
        current_test_fold = self.train_dataloader()
        input, gt_mask, file_name = batch
        print('filename: ', file_name)

        loss, z_hat_pred, accuracy = self.step(input, gt_mask)
        self.log(self.current_train_fold + "_" + "train_loss", loss, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)
        dict = self.create_return_dict('train', loss, input, file_name, gt_mask, z_hat_pred, self.current_train_fold)

        return {'loss': loss, 'accuracy': accuracy, 'dict': dict}


    def validation_step(self, batch, batch_idx):
        print('IN VALIDATION... ')

        input, gt_mask, file_name = batch
        f'filename: {file_name}'
        current_val_fold = file_name[0].rsplit("/", 3)[1]
        f'current_val_fold: {current_val_fold}'
        loss, z_hat_pred, accuracy = self.step(input, gt_mask)
        self.log("train_" + self.current_train_fold + "_" + "val_loss", loss, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)

        dict = self.create_return_dict('val', loss, input, file_name, gt_mask, z_hat_pred, current_val_fold)

        return dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.UnetModel.parameters(), lr=1e-4)
        return optimizer


    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        unet_module = parser.add_argument_group(
            title='tiramisu_module  specific args options')
        unet_module.add_argument("--batch_size", default=1, type=int)
        unet_module.add("--in_channels", default=1, type=int)
        unet_module.add("--out_channels", default=1, type=int)

        return parser
