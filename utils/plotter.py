import matplotlib.pyplot as plt
from matplotlib import gridspec
import plotly.express as px
from pytorch_lightning.callbacks import Callback
import torch
import torchvision
import wandb
import io
from PIL import Image, ImageFont, ImageDraw
import matplotlib.font_manager as fm # to create font
# from pympler import muppy, summary

EVERY_N = 5
THRESHOLD = 0.5
# top/last N items
N = 20


class Plotter(Callback):
    def __init__(self, current_train_fold):
        self.figs = []
        self.plot_figs = []
        self.train_fold_dict = []
        self.val_fold_dict = []
        self.val_fold_dict_epoch = []
        self.current_train_fold = current_train_fold
        self.plt_test_img_cnt = 0

    def log_image(self, plt, caption, trainer, pl_module):
        caption = "Trained on: " + self.current_train_fold + caption
        print('CAPTION: ', caption)
        pl_module.logger.experiment.log({caption: [wandb.Image(plt, caption=caption)]}, commit=True) #commit=True pushesh immediately to wandb

    def reshape_input(self, in_tensor):
        in_tensor = in_tensor[0, :, :, :].cpu()
        return in_tensor.numpy()

    def plot(self, imgs):
        imgs_reshaped = [self.reshape_input(i) for i in imgs]

        rows = 1
        cols = len(imgs)
        fig = plt.figure(figsize=(cols * 4, rows * 4))
        spec = gridspec.GridSpec(rows, cols, fig, wspace=0, hspace=0)
        spec.tight_layout(fig)
        for idx, img in enumerate(imgs_reshaped):
            plt.subplot(spec[0, idx]).imshow(img[0, :, :], cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0)
        # plt.savefig("../fig.png", bbox_inches='tight')
        # plt.show()

        return fig

    def preprocess_plot(self, file_name, orig_input, unet_output, gt_mask, result):
        file_name = file_name[0].rsplit("/", 1)[1]
        fontsize = 28
        ###############################################################################################################
        plot_fig = self.plot([orig_input, unet_output, gt_mask, result])
        buf = io.BytesIO()
        plot_fig.savefig(buf, format='png')
        buf.seek(0)
        plot_fig_pil = Image.open(buf)
        plot_fig_pil_draw = ImageDraw.Draw(plot_fig_pil)
        font = ImageFont.truetype(fm.findfont(fm.FontProperties()), fontsize)
        plot_fig_pil_draw.text((10, 10), file_name, (255, 255, 255), font=font)
        plot_fig_pil_t = torchvision.transforms.ToTensor()(plot_fig_pil)
        self.plot_figs.append(plot_fig_pil_t)
        plot_fig.clf()  #clear the figure
        plt.close(plot_fig)     #closes windows associated with the fig

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.train_fold_dict.append(outputs[0][0]['extra']['dict'])

    ###########################################################################################################
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.val_fold_dict.append(outputs)
        self.val_fold_dict_epoch.append(outputs)

    # def on_test_end(self, trainer, pl_module):
    def on_validation_epoch_end(self, trainer, pl_module):
        print('-------------ON VALIDATION END--------------')

        # all_objects = muppy.get_objects()
        # sum1 = summary.summarize(all_objects)
        # summary.print_(sum1)
        # Get references to certain types of objects such as dataframe
        # data = [ao for ao in all_objects if isinstance(ao, str)]
        # for d in data:
        #     print(d)
        #     print(len(d))

        folds_loss_stack = sorted(torch.stack([x['val_loss'] for x in self.val_fold_dict_epoch]))

        get_top_last = [folds_loss_stack[:N], folds_loss_stack[-N:]]
        caption = [f"best_{N}", f"worst_{N}"]
        for i in range(2):
            for x in self.val_fold_dict_epoch:
                if x['val_loss'] in get_top_last[i]:
                    self.preprocess_plot(x["file_name"], x["val_images_unet"][0], x["val_images_unet"][1],
                                        x["val_images_pred"][0], x["val_images_pred"][1])

            self.log_image(torchvision.utils.make_grid(self.plot_figs), caption[i] +
                           "|orig_input|unet_out|gt_mask|final_seg", trainer, pl_module)
            del self.plot_figs[:]

        del self.val_fold_dict_epoch[:]

    def on_train_end(self, trainer, pl_module):
        print('---------------ON TRAIN END--------------')
        best_model_epoch = torch.load(trainer.checkpoint_callback.best_model_path)['epoch']-1
        # best_model_loss = trainer.checkpoint_callback.best_model_score
        print('best_model_epoch: ', best_model_epoch)

        train_val_dict = [self.train_fold_dict, self.val_fold_dict]
        for i in range(2):
            plot_data = {'aorta_size': [], 'aorta_X_coord_mean': [], 'aorta_Y_coord_mean': [], 'DICE_loss': [],
                         'patient': []}
            type = 'train' if i == 0 else 'val'
            # print(train_val_dict[i])
            for fold_dict in train_val_dict[i]:
                if fold_dict['epoch'] == best_model_epoch:
                    plot_data['aorta_size'].append(torch.sum(fold_dict[f'{type}_images_pred'][0]))
                    plot_data['DICE_loss'].append(fold_dict[f'{type}_loss'].item())
                    plot_data['patient'].append(fold_dict[f'current_{type}_fold'])
                    indices_x_y = torch.nonzero(fold_dict[f'{type}_images_pred'][0][0, 0, :, :] == 1.0)
                    plot_data['aorta_X_coord_mean'].append(
                        (indices_x_y[:, 0] * 1.0).mean())  # mean x coord of all aorta pixels for this image
                    plot_data['aorta_Y_coord_mean'].append((indices_x_y[:, 1] * 1.0).mean())

            fig = px.scatter(plot_data, x='aorta_size', y='DICE_loss', color="patient")
            # fig.show()
            caption = 'During training' if i == 0 else 'Eval'
            pl_module.logger.experiment.log(
                {f"{caption} Aorta size/DICE loss (Trained on: {self.current_train_fold})": fig})
            #############################################################################################
            # plot_data['aorta_X_coord'] = torch.unbind(torch.unique(indices_x_y[:, 0]))   #unique removes duplicate values, unbind is the opposite of stack
            fig = px.scatter(plot_data, x='aorta_X_coord_mean', y='DICE_loss', color="patient")
            pl_module.logger.experiment.log(
                {f"{caption} Aorta X coord mean/DICE loss (Trained on: {self.current_train_fold})": fig})

            fig = px.scatter(plot_data, x='aorta_Y_coord_mean', y='DICE_loss', color="patient")
            pl_module.logger.experiment.log(
                {f"{caption} Aorta Y coord mean/DICE loss (Trained on: {self.current_train_fold})": fig})

        del self.train_fold_dict[:]
        del self.val_fold_dict[:]

    def unblockshaped(self, arr, h, w):
        """
        Return an array of shape (h, w) where
        h * w = arr.size

        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        n, nrows, ncols = arr.shape
        return (arr.reshape(h // nrows, -1, nrows, ncols)
                .swapaxes(1, 2)
                .reshape(h, w))