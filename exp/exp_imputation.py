import pandas as pd
from matplotlib import pyplot as plt

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.tools import results_evaluation, save_config
from utils.masking import mask_custom
from utils.metrics_imputation import calc_mae, calc_mse, results_evaluation_imputation, interpolate_nan_matrix

from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')

class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)
        if self.loss_method == "adaptive":
            self.log_ori_loss = nn.Parameter(torch.zeros(1))
            self.log_missing_loss = nn.Parameter(torch.zeros(1))
            self.log_forecast_loss = nn.Parameter(torch.zeros(1))


    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _select_scheduler(self, model_optim, train_loader):
        train_steps = len(train_loader)
        if self.args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        return scheduler

    def _loss_function(self, criterion, pred, true, x, mask):
        if self.args.loss == 'MSE':
            loss = calc_mse
        elif self.args.loss == 'MAE':
            loss = calc_mae
        else:
            raise NotImplementedError

        forecast_loss = criterion(pred,true)

        if self.args.loss_method == "fix":
            ori_loss = loss(x, true, mask)
            missing_loss = loss(x, true, mask ^ 1)
            loss = self.args.ori_weight * ori_loss + self.args.missing_weight * missing_loss + self.args.forecast_weight * forecast_loss

        elif self.args.loss_method == "adaptive":
            if self.args.model == 'TimeLLMformer':
                missing_loss = 0
                for i, tensor in enumerate(x):
                    missing_loss += loss(tensor, true, mask)
                    if i == len(x) - 1:  # 仅在最后一个 tensor 时计算 MIT_loss
                        ori_loss = self.args.ori_weight * self.loss_func(tensor, true, mask ^ 1)
                missing_loss = self.args.missing_weight * missing_loss / len(x)
            else:
                ori_loss = loss(x, true, mask)
                missing_loss = loss(x, true, mask ^ 1)

            loss = 0.5 * (torch.exp(-self.log_ori_loss.to(true.device)) * ori_loss + torch.exp(-self.log_missing_loss.to(true.device)) * missing_loss + torch.exp(-self.log_forecast_loss.to(true.device)) * forecast_loss +
                              self.log_ori_loss.to(true.device) + self.log_missing_loss.to(true.device) + self.log_forecast_loss.to(true.device))

        return loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                x_forecast = x_forecast.float().to(self.device)
                if self.args.mask_target_only:
                    inp = mask_custom(batch_x[:, :, -self.f_dim:], mask_rate=self.args.mask_rate, method='rdo')
                else:
                    inp = mask_custom(batch_x, mask_rate=self.args.mask_rate, method='rdo')
                mask = np.isnan(inp) ^ np.isnan(batch_x)
                inp = inp.float().to(self.device)
                mask = mask.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast,mask=mask)[0]
                        else:
                            outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast,mask=mask)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast,mask=mask)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, x_forecast,mask=mask)

                if self.args.accelerate:
                    outputs, batch_y = self.accelerator.gather_for_metrics((outputs, batch_y))
                outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                mask = mask.detach().cpu()
                if self.args.model != 'TimeLLMformer':
                    outputs = outputs.detach().cpu()
                else:
                    outputs = tuple(tensor.detach().cpu() for tensor in outputs)

                loss = self._loss_function(criterion, pred, true, outputs, mask)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting, 'checkpoints')
        if not os.path.exists(path):
            os.makedirs(path)
        save_config(self.args, os.path.join(path, 'configs.pkl'))

        time_now = time.time()
        time_start = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,accelerator=self.accelerator)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(model_optim, train_loader)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        if self.args.accelerate:
            self.model,train_loader,vali_loader, model_optim,scheduler = self.accelerator.prepare(self.model,train_loader,vali_loader,model_optim,scheduler)
            self.accelerator.print(f"Process {self.accelerator.process_index} is using device {self.accelerator.device}")

        # Initialize a dictionary to store loss values
        loss_records = {"epoch": [], "time": [],"train_loss": [], "vali_loss": []}

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()
                x_forecast = x_forecast.float()
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                # imputation input
                if self.args.mask_target_only:
                    inp = mask_custom(batch_x[:, :, -self.f_dim:], mask_rate=self.args.mask_rate, method='rdo')
                else:
                    inp = mask_custom(batch_x, mask_rate=self.args.mask_rate, method='rdo')
                mask = np.isnan(inp) ^ np.isnan(batch_x)
                inp = inp.float()
                mask = mask.float()

                if self.args.accelerate:
                    pass

                else:
                    batch_y = batch_y.to(self.device)
                    batch_x_mark = batch_x_mark.to(self.device)
                    batch_y_mark = batch_y_mark.to(self.device)
                    x_forecast = x_forecast.to(self.device)
                    dec_inp = dec_inp.to(self.device)
                    inp = inp.to(self.device)
                    mask = mask.to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)[0]
                        else:
                            outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)

                        outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]
                        if self.args.accelerate:
                            batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:]
                        else:
                            batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)
                        loss = self._loss_function(criterion, outputs, batch_y, outputs, mask)

                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)[0]
                    else:
                        outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)

                    outputs = outputs[:, -self.args.pred_len:, -self.f_dim:]
                    if self.args.accelerate:
                        batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:]
                    else:
                        batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)
                    loss = self._loss_function(criterion, outputs, batch_y, outputs, mask)

                    train_loss.append(loss.item())
                if self.args.accelerate:
                    self.accelerator.print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.2f}min'.format(speed, left_time / 60))
                else:
                    verbose_interval = (len(train_loader) // 5) if len(train_loader) > 5 else 1
                    if (i + 1) % verbose_interval == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.2f}min'.format(speed, left_time / 60))
                        iter_count = 0
                        time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if self.args.accelerate:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()
                        model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False,accelerator=self.accelerator)
                    scheduler.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            # Record loss values
            loss_records["epoch"].append(epoch + 1)
            loss_records["time"].append(round((time.time() - time_start)/60,4))
            loss_records["train_loss"].append(train_loss)
            loss_records["vali_loss"].append(vali_loss)

            # test_loss = self.vali(test_data, test_loader, criterion)
            cost_time = round((time.time() - epoch_time) / 60, 2)
            print(" Epoch: {} cost time: {} min".format(epoch + 1, cost_time))
            print("☆☆☆☆☆Train Loss: {0:.7f} Vali Loss: {1:.7f}".format(train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            left_time = 1 + (self.args.patience - early_stopping.counter) * cost_time
            print("  Left time: {} min".format(round(left_time,2)))

            if self.args.lradj != 'TST':
                if self.args.lradj == 'COS':
                    scheduler.step()
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    # if epoch == 0:
                    #     self.args.learning_rate = model_optim.param_groups[0]['lr']
                    #     print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)

            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        if self.args.accelerate:
            self.accelerator.wait_for_everyone()

        best_model_path = path + '/' + 'checkpoint'
        if self.args.accelerate:
            self.model = self.accelerator.load_state(best_model_path)
        else:
            self.model.load_state_dict(torch.load(best_model_path))

        # Convert loss records to DataFrame and save as CSV
        folder_path = os.path.join(self.args.checkpoints, setting)
        loss_df = pd.DataFrame(loss_records)
        loss_df.to_csv(os.path.join(folder_path, "loss_records.csv"), index=False)
        print("Loss records saved to:", os.path.join(folder_path, "loss_records.csv"))
        report = torch.cuda.memory_summary(device=self.device, abbreviated=False)
        print(report)
        peak_alloc = torch.cuda.max_memory_allocated(self.device)
        used_bytes = torch.cuda.memory_allocated(self.device)
        with open(os.path.join(folder_path,"memory_summary_{}_{}.txt".format(round(used_bytes*1024/(10**9),1),round(speed*1000,2))), "w") as f:
            f.write(report)
        print("Saved CUDA memory summary to cuda_memory_summary.txt")
        return self.model

    def test(self, setting, test=0, path=None):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            if path is None:
                model_path = os.path.join(self.args.checkpoints, setting, 'checkpoints')
                folder_path = os.path.join(self.args.checkpoints, setting)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            else:
                model_path = os.path.join(path, 'checkpoints')
                folder_path = path
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint')))
        else:
            folder_path = os.path.join(self.args.checkpoints, setting)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        imputation_trues = []
        pred_trues = []
        imputations = []
        preds = []
        x_withnans = []
        masks = []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                x_forecast = x_forecast.float().to(self.device)

                if self.args.mask_target_only:
                    inp = mask_custom(batch_x[:, :, -self.f_dim:], mask_rate=self.args.mask_rate, method='rdo')
                else:
                    inp = mask_custom(batch_x, mask_rate=self.args.mask_rate, method='rdo')
                mask = np.isnan(inp) ^ np.isnan(batch_x)
                inp = inp.float().to(self.device)
                mask = mask.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)[0]
                        else:
                            outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)
                else:
                    if self.args.output_attention:
                        outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)[0]

                    else:
                        outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)

                if self.args.accelerate:
                    self.accelerator.wait_for_everyone()
                    outputs = self.accelerator.gather_for_metrics(outputs)

                inp = outputs[:,:self.args.seq_len, -self.f_dim:]
                mask = outputs[:,:self.args.seq_len, -self.f_dim:]
                outputs = outputs[:, :, -self.f_dim:]
                batch_x = batch_x[:, :self.args.seq_len, -self.f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:]  # .to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                x_withnans.append(inp)
                masks.append(mask)
                imputation_trues.append(batch_x)
                pred_trues.append(batch_y)
                imputations.append(outputs[:, :self.args.seq_len, :])
                preds.append(outputs[:, -self.args.pred_len:, :])

        x_withnans = np.concatenate(x_withnans, axis=0)
        preds = np.concatenate(preds, axis=0)
        masks = np.concatenate(masks, axis=0)
        imputation_trues = np.concatenate(imputation_trues, axis=0)
        pred_trues = np.concatenate(pred_trues, axis=0)
        imputations = np.concatenate(imputations, axis=0)

        print('test shape:', preds.shape, imputation_trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        pred_trues = pred_trues.reshape(-1, pred_trues.shape[-2], pred_trues.shape[-1])
        imputations = imputations.reshape(-1, imputations.shape[-2], imputations.shape[-1])
        imputation_trues = imputation_trues.reshape(-1, imputation_trues.shape[-1])
        print('test shape:', preds.shape, imputations.shape)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(imputations.shape[0]):
                x = imputations[i].reshape(-1, 1)
                y = imputation_trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        [mse, rmse,nrmse, mae,mape,rae, r2,corr] = results_evaluation(imputation_trues.flatten(), imputations.flatten())
        print('mae:{}, r2:{}, dtw:{}'.format(mae, r2, dtw))
        f = open(os.path.join('./results', "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, r2:{}, dtw:{}'.format(mae, r2, dtw))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(os.path.join(folder_path, 'metrics_{}_{}.npy'.format(self.args.data,self.args.data_path[:-4])), np.array([mae, mse, rmse, r2, corr]))
        np.save(os.path.join(folder_path, 'preds_{}_{}.npy'.format(self.args.data,self.args.data_path[:-4])), preds)
        np.save(os.path.join(folder_path, 'pred_trues_{}_{}.npy'.format(self.args.data,self.args.data_path[:-4])), pred_trues)
        np.save(os.path.join(folder_path, 'imputations_{}_{}.npy'.format(self.args.data,self.args.data_path[:-4])), imputations)
        np.save(os.path.join(folder_path, 'imputation_trues_{}_{}.npy'.format(self.args.data,self.args.data_path[:-4])), imputation_trues)

        if self.args.features == 'M':
            pred_res, metrics_df, imputation_metrics_df, interpolation_metrics_df = self.res_evaluation_multi_target(imputation_trues, imputations, x_withnans, masks,
                                                                                                                     trainable_params, self.folder_path)
        else:
            pred_res, metrics_df = self.res_evaluation(imputation_trues, imputations, x_withnans, masks, trainable_params, self.folder_path)
        return pred_res,metrics_df

    def res_evaluation(self, X_ori, pred, X_withnan, indicating_mask, trainable_params, path):

        stride = self.args.pred_len
        pred_output = np.squeeze(pred, axis=-1)[::stride, :].reshape(-1, 1)
        true_output = np.squeeze(X_ori, axis=-1)[::stride, :].reshape(-1, 1)
        X_withnan = np.squeeze(X_withnan, axis=-1)[::stride, :].reshape(-1, 1)
        indicating_mask = np.squeeze(indicating_mask, axis=-1)[::stride, :].reshape(-1, 1)

        pred_interpolate = interpolate_nan_matrix(X_withnan.reshape(-1, 1), method=self.args.interpolate_method, order=self.args.interpolate_order)
        nan_mask = np.isnan(pred_interpolate)
        pred_interpolate[nan_mask] = true_output[nan_mask]
        # if np.isnan(pred_interpolate[0,0]):
        #     pred_interpolate[0, 0] = true_output[0, 0]

        pred_res = pd.DataFrame(
            {'X_ori': true_output.flatten(), 'X_pred': pred_output.flatten(), 'X_pred_interpolate': pred_interpolate.flatten(), 'X_withnan': X_withnan.flatten()})
        pred_res.loc[pred_res['X_ori'] < 1e-3, 'X_pred'] = 0
        pred_res.loc[pred_res['X_ori'] < 1e-3, 'X_pred_interpolate'] = 0
        pred_res.loc[pred_res['X_ori'] < 1e-3, 'X_ori'] = 0
        pred_res.loc[pred_res['X_ori'] < 1e-3, 'X_withnan'] = 0

        [mse, rmse, nrmse, mae, mape, rae, r2, corr] = results_evaluation(pred_res['X_ori'].values, pred_res['X_pred'].values)
        [mse_imputation, rmse_imputation, mae_imputation, mre_imputation] = results_evaluation_imputation(np.nan_to_num(X_withnan), pred_output, indicating_mask)
        [mse_imputation_inter, rmse_imputation_inter, mae_imputation_inter, mre_imputation_inter] = results_evaluation_imputation(np.nan_to_num(X_withnan), pred_interpolate,
                                                                                                                                  indicating_mask)

        pred_res.to_csv(os.path.join(path, 'pred_results_{}_{}.csv'.format(self.args.data, self.args.data_path[:-4])))
        metrics_df = pd.DataFrame({'trainable_params': trainable_params, 'mse': mse, 'rmse': rmse, 'nrmse': nrmse, 'mae': mae, 'mape': mape, 'rae': rae, 'r2': r2, 'corr': corr},
                                  index=[0])
        metrics_df.to_csv(os.path.join(path, 'metrics_results_{}_{}.csv'.format(self.args.data, self.args.data_path[:-4])))

        metrics_df_imputation = pd.DataFrame({'mse_imputation': mse_imputation, 'rmse_imputation': rmse_imputation, 'mae_imputation': mae_imputation,
                                              'mre_imputation': mre_imputation, 'mse_imputation_inter': mse_imputation_inter, 'rmse_imputation_inter': rmse_imputation_inter,
                                              'mae_imputation_inter': mae_imputation_inter, 'mre_imputation_inter': mre_imputation_inter, }, index=[0])
        metrics_df_imputation.to_csv(os.path.join(path, 'metrics_results_imputation_{}_{}.csv'.format(self.args.data, self.args.data_path[:-4])))

        print('MAE: {} NAE_inter: {}'.format(mae_imputation, mae_imputation_inter))
        return pred_res, metrics_df

    def res_evaluation_multi_target(self, X_ori, pred, X_withnan, indicating_mask, trainable_params, path):
        stride = self.args.pred_len
        true = X_ori[::stride, :, :].reshape(-1, len(self.args.target))
        pred = pred[::stride, :, :].reshape(-1, len(self.args.target))
        X_withnan = X_withnan[::stride, :, :].reshape(-1, len(self.args.target))
        indicating_mask = indicating_mask[::stride, :, :].reshape(-1, len(self.args.target))
        pred_interpolate = interpolate_nan_matrix(X_withnan, method=self.args.interpolate_method, order=self.args.interpolate_order)

        # Create DataFrame for true and predicted values
        columns = [f"{i}_{col}" for i in self.args.target for col in ["ori", "pred", "pred_inter", "X_withnan"]]
        res_df = pd.DataFrame(np.hstack([true, pred, pred_interpolate, X_withnan]), columns=columns)

        # Initialize metrics dictionaries
        metrics = {key: [] for key in ["mse", "rmse", "nrmse", "mae", "mape", "rae", "r2", "corr"]}
        imputation_metrics = {key: [] for key in ["mse_imputation", "rmse_imputation", "mae_imputation", "mre_imputation"]}
        interpolation_metrics = {key: [] for key in ["mse_imputation_inter", "rmse_imputation_inter", "mae_imputation_inter", "mre_imputation_inter"]}

        # Calculate metrics for each target
        for idx, target in enumerate(self.args.target):
            t_true, t_pred, t_pred_inter, t_withnan, t_mask = (
                true[:, idx], pred[:, idx], pred_interpolate[:, idx], X_withnan[:, idx], indicating_mask[:, idx])

            # Metrics for full sequence
            mse, rmse, nrmse, mae, mape, rae, r2, corr = results_evaluation(t_true, t_pred)
            metrics["mse"].append(mse)
            metrics["rmse"].append(rmse)
            metrics["nrmse"].append(nrmse)
            metrics["mae"].append(mae)
            metrics["mape"].append(mape)
            metrics["rae"].append(rae)
            metrics["r2"].append(r2)
            metrics["corr"].append(corr)

            # Metrics for imputation (model output)
            mse_imp, rmse_imp, mae_imp, mre_imp = results_evaluation_imputation(
                np.nan_to_num(t_withnan), t_pred, t_mask)

            imputation_metrics["mse_imputation"].append(mse_imp)
            imputation_metrics["rmse_imputation"].append(rmse_imp)
            imputation_metrics["mae_imputation"].append(mae_imp)
            imputation_metrics["mre_imputation"].append(mre_imp)

            # Metrics for imputation (interpolation)
            mse_imp_inter, rmse_imp_inter, mae_imp_inter, mre_imp_inter = results_evaluation_imputation(
                np.nan_to_num(t_withnan), t_pred_inter, t_mask)

            interpolation_metrics["mse_imputation_inter"].append(mse_imp_inter)
            interpolation_metrics["rmse_imputation_inter"].append(rmse_imp_inter)
            interpolation_metrics["mae_imputation_inter"].append(mae_imp_inter)
            interpolation_metrics["mre_imputation_inter"].append(mre_imp_inter)

        # Create DataFrames for metrics
        metrics_df = pd.DataFrame(metrics, index=self.args.target)
        imputation_metrics_df = pd.DataFrame(imputation_metrics, index=self.args.target)
        interpolation_metrics_df = pd.DataFrame(interpolation_metrics, index=self.args.target)

        # Add mean row
        metrics_df.loc["mean"] = metrics_df.mean()
        imputation_metrics_df.loc["mean"] = imputation_metrics_df.mean()
        interpolation_metrics_df.loc["mean"] = interpolation_metrics_df.mean()
        print(imputation_metrics_df.loc["mean"])

        # Save DataFrames
        res_df.to_csv(os.path.join(path, f"pred_res_{self.args.data_path[:-4]}.csv"))
        metrics_df.to_csv(os.path.join(path, f"metrics_df_{self.args.data_path[:-4]}.csv"))
        imputation_metrics_df.to_csv(os.path.join(path, f"imputation_metrics_df_{self.args.data_path[:-4]}.csv"))
        interpolation_metrics_df.to_csv(os.path.join(path, f"interpolation_metrics_df_{self.args.data_path[:-4]}.csv"))

        return res_df, metrics_df, imputation_metrics_df, interpolation_metrics_df

    def _show_plot(self,i,y_true,y_pred,path=None):
        x_range = np.arange(self.args.num_train -self.args.pred_len*7, self.args.num_train)
        y_pred_plot = y_pred[-self.args.pred_len*7:]
        plt.figure(self.args.target.index(i)+1, figsize=(20, 5))
        plt.plot(x_range, y_pred_plot, "r-", label="Forecast values")
        yplot = y_true[-self.args.pred_len*7:]
        plt.plot(x_range, yplot, "k-", label="True values")
        ymin, ymax = plt.ylim()
        plt.vlines(self.args.num_train - self.args.pred_len*7, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        plt.ylim(ymin, ymax)
        plt.legend(loc="upper left")
        plt.title('Prediction')
        plt.xlabel("Periods")
        plt.ylabel("Y")
        plt.savefig(os.path.join(path,'{}.png'.format(i)))
        plt.close()
