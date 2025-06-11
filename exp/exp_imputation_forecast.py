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


class Exp_Imputation_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation_Forecast, self).__init__(args)
        if self.args.loss_method == "adaptive":
            self.log_missing_loss = nn.Parameter(torch.zeros(1))
            self.log_pred_loss = nn.Parameter(torch.zeros(1))
        if self.args.loss == 'MSE':
            self.loss_func = calc_mse
        elif self.args.loss == 'MAE':
            self.loss_func = calc_mae
        else:
            raise NotImplementedError

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        params = list(self.model.parameters())
        if self.args.loss_method == "adaptive":
            # … plus your two learnable scalars
            params += [self.log_pred_loss, self.log_missing_loss]
        model_optim = optim.Adam(params, lr=self.args.learning_rate)
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

    def _loss_function(self, outputs, true, mask):
        if isinstance(outputs, tuple):
            pred = outputs[-1][:, -self.args.pred_len:, :]
            imputation = tuple(
                o[:, :self.args.seq_len, -self.f_dim:]
                for o in outputs
            )
        else:
            pred = outputs[:, -self.args.pred_len:, :]
            imputation = outputs[:, :self.args.seq_len, :]

        pred_true = true[:, -self.args.pred_len:, :]
        imputation_true = true[:, :self.args.seq_len, :]
        pred_loss = self.loss_func(pred, pred_true)

        if self.args.model == 'TimeLLMformer':
            missing_loss = 0
            for i, tensor in enumerate(imputation):
                missing_loss += self.loss_func(tensor, imputation_true, mask ^ 1)
            missing_loss = self.args.missing_weight * missing_loss / len(imputation)
        else:
            missing_loss = self.loss_func(imputation, imputation_true, mask ^ 1)

        if self.args.loss_method == "fix":
            loss = self.args.missing_weight * missing_loss + self.args.pred_weight * pred_loss

        elif self.args.loss_method == "adaptive":
            loss = 0.5 * (torch.exp(-self.log_missing_loss.to(true.device)) * missing_loss + torch.exp(
                -self.log_pred_loss.to(true.device)) * pred_loss + self.log_missing_loss.to(true.device) + self.log_pred_loss.to(true.device))

        return loss

    def vali(self, vali_data, vali_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast) in enumerate(vali_loader):

                if self.args.mask_target_only:
                    inp = mask_custom(batch_x[:, :, -self.f_dim:], mask_rate=self.args.mask_rate, method='rdo')
                else:
                    inp = mask_custom(batch_x, mask_rate=self.args.mask_rate, method='rdo')
                mask = (np.isnan(inp) ^ np.isnan(batch_x)) ^ 1
                inp = batch_x.masked_fill(mask == 0, 0)
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                x_forecast = x_forecast.float().to(self.device)
                inp = inp.float().to(self.device)
                mask = mask.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)[0]
                else:
                    outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)

                if self.args.accelerate:
                    outputs, batch_y = self.accelerator.gather_for_metrics((outputs, batch_y))
                if isinstance(outputs, tuple):
                    outputs = tuple(
                        o[:, :self.args.seq_len + self.args.pred_len, -self.f_dim:].detach().cpu()
                        for o in outputs
                    )
                else:
                    outputs = outputs[:, :self.args.seq_len + self.args.pred_len, -self.f_dim:]
                    outputs = outputs.detach().cpu()
                true = torch.cat([batch_x[:, :self.args.seq_len, -self.f_dim:], batch_y[:, -self.args.pred_len:, -self.f_dim:]], dim=1).detach().cpu()
                mask = mask.detach().cpu()

                loss = self._loss_function(outputs, true, mask[:, :self.args.seq_len, -self.f_dim:])

                total_loss.append(loss.item())
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
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, accelerator=self.accelerator)

        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim, train_loader)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        if self.args.accelerate:
            self.model, train_loader, vali_loader, model_optim, scheduler = self.accelerator.prepare(self.model, train_loader, vali_loader, model_optim, scheduler)
            self.accelerator.print(f"Process {self.accelerator.process_index} is using device {self.accelerator.device}")

        # Initialize a dictionary to store loss values
        loss_records = {"epoch": [], "time": [], "train_loss": [], "vali_loss": []}
        if self.args.loss_method == 'adaptive':
            loss_records = {"epoch": [],  "time": [],"train_loss": [], "vali_loss": [], "missing_weight": [], "pred_weight": []}

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
                mask = (np.isnan(inp) ^ np.isnan(batch_x)) ^ 1
                inp = batch_x.masked_fill(mask == 0, 0)

                if self.args.accelerate:
                    pass

                else:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_x_mark = batch_x_mark.to(self.device)
                    batch_y_mark = batch_y_mark.to(self.device)
                    x_forecast = x_forecast.to(self.device)
                    dec_inp = dec_inp.to(self.device)
                    inp = inp.to(self.device)
                    mask = mask.to(self.device)

                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)[0]
                else:
                    outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)

                if isinstance(outputs, tuple):
                    outputs = tuple(
                        o[:, :self.args.seq_len + self.args.pred_len, -self.f_dim:]
                        for o in outputs
                    )
                else:
                    outputs = outputs[:, :self.args.seq_len + self.args.pred_len, -self.f_dim:]
                if self.args.accelerate:
                    batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:]
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, -self.f_dim:].to(self.device)
                trues = torch.cat([batch_x[:, :self.args.seq_len, -self.f_dim:], batch_y[:, -self.args.pred_len:, -self.f_dim:]], dim=1)
                loss = self._loss_function(outputs, trues, mask[:, :self.args.seq_len, -self.f_dim:])

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
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False, accelerator=self.accelerator)
                    scheduler.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)

            # Record loss values
            loss_records["epoch"].append(epoch + 1)
            loss_records["time"].append(round((time.time() - time_start) / 60, 4))
            loss_records["train_loss"].append(train_loss)
            loss_records["vali_loss"].append(vali_loss)
            if self.args.loss_method == 'adaptive':
                loss_records["missing_weight"].append(self.log_missing_loss.detach().numpy())
                loss_records["pred_weight"].append(self.log_pred_loss.detach().numpy())

            # test_loss = self.vali(test_data, test_loader, criterion)
            cost_time = round((time.time() - epoch_time) / 60, 2)
            print(" Epoch: {} cost time: {} min".format(epoch + 1, cost_time))
            print("☆☆☆☆☆Train Loss: {0:.7f} Vali Loss: {1:.7f}".format(train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            left_time = 1 + (self.args.patience - early_stopping.counter) * cost_time
            print("  Left time: {} min".format(round(left_time, 2)))

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
        with open(os.path.join(folder_path, "memory_summary_{}_{}.txt".format(round(used_bytes * 1024 / (10 ** 9), 1), round(speed * 1000, 2))), "w") as f:
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

        preds, trues, masks = [], [], []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, x_forecast) in enumerate(test_loader):

                if self.args.mask_target_only:
                    inp = mask_custom(batch_x[:, :, -self.f_dim:], mask_rate=self.args.mask_rate, method='rdo')
                else:
                    inp = mask_custom(batch_x, mask_rate=self.args.mask_rate, method='rdo')
                mask = (np.isnan(inp) ^ np.isnan(batch_x)) ^ 1
                inp = batch_x.masked_fill(mask == 0, 0)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                x_forecast = x_forecast.float().to(self.device)
                inp = inp.float().to(self.device)
                mask = mask.to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)[0]

                else:
                    outputs = self.model(inp, batch_x_mark, dec_inp, batch_y_mark, x_forecast, mask=mask)

                if self.args.accelerate:
                    self.accelerator.wait_for_everyone()
                    outputs = self.accelerator.gather_for_metrics(outputs)

                if isinstance(outputs, tuple):
                    outputs = outputs[-1][:, :self.args.seq_len + self.args.pred_len, -self.f_dim:]
                else:
                    outputs = outputs[:, :self.args.seq_len + self.args.pred_len, -self.f_dim:]
                mask = mask[:, :, -self.f_dim:]
                true = torch.cat([batch_x[:, :self.args.seq_len, -self.f_dim:], batch_y[:, -self.args.pred_len:, -self.f_dim:]], dim=1)
                outputs = outputs.detach().cpu().numpy()
                true = true.detach().cpu().numpy()
                mask = mask.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    true = test_data.inverse_transform(true.reshape(shape[0] * shape[1], -1)).reshape(shape)

                masks.append(mask)
                trues.append(true)
                preds.append(outputs)

        preds = np.concatenate(preds, axis=0)
        masks = np.concatenate(masks, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        [mse, rmse, nrmse, mae, mape, rae, r2, corr] = results_evaluation(trues.flatten(), preds.flatten())
        print('mae:{}, r2:{}, dtw:{}'.format(mae, r2, dtw))
        f = open(os.path.join('./results', "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, r2:{}, dtw:{}'.format(mae, r2, dtw))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(os.path.join(folder_path, 'metrics_{}_{}.npy'.format(self.args.data, self.args.data_path[:-4])), np.array([mae, mse, rmse, r2, corr]))
        np.save(os.path.join(folder_path, 'preds_{}_{}.npy'.format(self.args.data, self.args.data_path[:-4])), preds)
        np.save(os.path.join(folder_path, 'trues_{}_{}.npy'.format(self.args.data, self.args.data_path[:-4])), trues)

        pred_res, metrics_df, imputation_metrics_df = self.res_evaluation_multi_target(preds, trues, masks, trainable_params, folder_path)
        return pred_res, metrics_df, imputation_metrics_df

    def res_evaluation_multi_target(self, pred, true, mask, trainable_params, path):
        stride = self.args.pred_len + self.args.seq_len

        imputation_true = true[:,:self.args.seq_len,:][::stride, :, :].reshape(-1, len(self.args.target))
        imputation = pred[:,:self.args.seq_len,:][::stride, :, :].reshape(-1, len(self.args.target))
        pred_true = true[:,-self.args.pred_len:,:][::stride, :, :].reshape(-1, len(self.args.target))
        pred = pred[:,-self.args.pred_len:,:][::stride, :, :].reshape(-1, len(self.args.target))
        mask = mask[::stride, :, :].reshape(-1, len(self.args.target))
        X_withnan = np.where(mask == 0, np.nan, imputation_true)

        mask_int = mask.astype(int)  # 转成整数 0/1
        mask = mask_int ^ 1  # 0↔1 取反
        interpolate = interpolate_nan_matrix(X_withnan, method=self.args.interpolate_method, order=self.args.interpolate_order)

        # Create DataFrame for true and predicted values
        # columns = [f"{i}_{col}" for col in ["imputation_true", "imputation","interpolate", "withnan","pred_true", "pred"] for i in self.args.target]
        # res_df = pd.DataFrame(np.hstack([imputation_true, imputation,interpolate,X_withnan, pred_true, pred]), columns=columns)
        interp_columns = [f"{i}_{col}" for i in self.args.target for col in ["imputation_true", "imputation", "interpolate", "withnan"]]
        data_blocks = [imputation_true, imputation, interpolate, X_withnan]
        data_parts = [np.hstack([block[:, i].reshape(-1, 1) for block in data_blocks]) for i in range(len(self.args.target))]
        interp_df = pd.DataFrame(np.hstack(data_parts), columns=interp_columns)

        # 生成预测结果 DataFrame
        pred_columns = [f"{i}_{col}" for i in self.args.target for col in ["pred_true", "pred"]]
        data_blocks = [pred_true, pred]
        data_parts = [np.hstack([block[:, i].reshape(-1, 1) for block in data_blocks]) for i in range(len(self.args.target))]
        pred_df = pd.DataFrame(np.hstack(data_parts), columns=pred_columns)

        # 为两个 DataFrame 添加合适的 index（以便对齐或后续合并）
        interp_df.index.name = "imputation_index"
        pred_df.index.name = "prediction_index"
        res_df = pd.concat([interp_df, pred_df], axis=0, ignore_index=False)

        # Initialize metrics dictionaries
        metrics = {key: [] for key in ["trainable_params", "mse", "rmse", "nrmse", "mae", "mape", "rae", "r2", "corr"]}
        imputation_metrics = {key: [] for key in ["trainable_params", "mse_imputation", "rmse_imputation", "mae_imputation", "mre_imputation","mse_imputation_inter", "rmse_imputation_inter", "mae_imputation_inter", "mre_imputation_inter"]}

        # Calculate metrics for each target
        for idx, target in enumerate(self.args.target):
            imputation_true_i, pred_true_i, imputation_i, pred_i, interpolate_i, withnan_i, mask_i = imputation_true[:, idx], pred_true[:, idx], imputation[:, idx], pred[:, idx], interpolate[:,idx], X_withnan[:,idx], mask[:,idx]

            # Metrics for full sequence
            mse, rmse, nrmse, mae, mape, rae, r2, corr = results_evaluation(pred_true_i, pred_i)
            metrics['trainable_params'] = trainable_params
            metrics["mse"].append(mse)
            metrics["rmse"].append(rmse)
            metrics["nrmse"].append(nrmse)
            metrics["mae"].append(mae)
            metrics["mape"].append(mape)
            metrics["rae"].append(rae)
            metrics["r2"].append(r2)
            metrics["corr"].append(corr)

            # Metrics for imputation (model output)
            mse_imp, rmse_imp, mae_imp, mre_imp = results_evaluation_imputation(imputation_true_i, imputation_i, mask_i)
            mse_imp_inter, rmse_imp_inter, mae_imp_inter, mre_imp_inter = results_evaluation_imputation(imputation_true_i, interpolate_i, mask_i)
            imputation_metrics['trainable_params'] = trainable_params
            imputation_metrics["mse_imputation"].append(mse_imp)
            imputation_metrics["rmse_imputation"].append(rmse_imp)
            imputation_metrics["mae_imputation"].append(mae_imp)
            imputation_metrics["mre_imputation"].append(mre_imp)
            imputation_metrics["mse_imputation_inter"].append(mse_imp_inter)
            imputation_metrics["rmse_imputation_inter"].append(rmse_imp_inter)
            imputation_metrics["mae_imputation_inter"].append(mae_imp_inter)
            imputation_metrics["mre_imputation_inter"].append(mre_imp_inter)


        # Create DataFrames for metrics
        metrics_df = pd.DataFrame(metrics, index=self.args.target)
        imputation_metrics_df = pd.DataFrame(imputation_metrics, index=self.args.target)

        # Add mean row
        metrics_df.loc["mean"] = metrics_df.mean()
        imputation_metrics_df.loc["mean"] = imputation_metrics_df.mean()
        print(imputation_metrics_df.loc["mean"])
        print(metrics_df.loc["mean"])
        # Save DataFrames
        res_df.to_csv(os.path.join(path, f"pred_res_{self.args.data_path[:-4]}.csv"))
        metrics_df.to_csv(os.path.join(path, f"metrics_df_{self.args.data_path[:-4]}.csv"))
        imputation_metrics_df.to_csv(os.path.join(path, f"imputation_metrics_df_{self.args.data_path[:-4]}.csv"))
        return res_df, metrics_df, imputation_metrics_df,

    def _show_plot(self, i, y_withnan, y_true, y_imputation, y_inter, path=None):
        x_range = np.arange(self.args.num_train - self.args.pred_len * 7, self.args.num_train)
        y_true_plot = y_true[-self.args.pred_len * 7:]
        plt.figure(self.args.target.index(i) + 1, figsize=(20, 5))
        plt.plot(x_range, y_true_plot, "r-", label="True values")

        y_withnan_plot = y_withnan[-self.args.pred_len * 7:]
        plt.plot(x_range, y_withnan_plot, "k-", label="With nan values")

        y_imputation_plot = y_imputation[-self.args.pred_len * 7:]
        plt.plot(x_range, y_imputation_plot, "g-", linestyles="dashed", label="Imputation values")

        y_inter_plot = y_inter[-self.args.pred_len * 7:]
        plt.plot(x_range, y_inter_plot, "b-", linestyles="dashed", label="Interplote values")
        ymin, ymax = plt.ylim()
        plt.vlines(self.args.num_train - self.args.pred_len * 7, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        plt.ylim(ymin, ymax)
        plt.legend(loc="upper left")
        plt.title('Prediction')
        plt.xlabel("Periods")
        plt.ylabel("Y")
        plt.savefig(os.path.join(path, '{}.png'.format(i)))
        plt.close()
