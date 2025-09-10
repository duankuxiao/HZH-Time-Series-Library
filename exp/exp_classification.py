import pandas as pd
from matplotlib import pyplot as plt

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, cal_accuracy
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
from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)

        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, padding_mask, None, None)
                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='test')
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
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())
                verbose_interval = (len(train_loader) // 5) if len(train_loader) > 5 else 1
                if (i + 1) % verbose_interval == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.2f}min'.format(speed, left_time / 60))
                    iter_count = 0
                    time_now = time.time()

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    model_optim.step()

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

        best_model_path = path + '/' + 'checkpoint'

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

        preds = []
        trues = []

        self.model.eval()
        time_now = time.time()
        cost_time = []

        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)
                speed = round((time.time() - time_now), 4)
                time_now = time.time()

        cost_time.append(speed)
        print("Cost time: {} s/iter".format(round(np.mean(cost_time),4)))

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        [mse, rmse,nrmse, mae,mape,rae, r2,corr] = results_evaluation(trues.flatten(), preds.flatten())
        print('accuracy:{}'.format(accuracy))
        f = open(os.path.join('./results', "result_classification.txt"), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(os.path.join(folder_path, 'metrics_{}_{}.npy'.format(self.args.data,self.args.data_path[:-4])), np.array([mae, mse, rmse, r2, corr]))
        np.save(os.path.join(folder_path, 'pred_{}_{}.npy'.format(self.args.data,self.args.data_path[:-4])), preds)
        np.save(os.path.join(folder_path, 'true_{}_{}.npy'.format(self.args.data,self.args.data_path[:-4])), trues)

        if self.args.features == 'M':
            pred_res,metrics_df = self.res_evaluation_multi_target(trues, preds, trainable_params, folder_path)
        else:
            pred_res,metrics_df = self.res_evaluation(trues,preds,trainable_params, folder_path)
        return pred_res,metrics_df

    def res_evaluation(self,true, pred,trainable_params, path):
        stride = self.args.pred_len
        pred_output = np.squeeze(pred,axis=-1)[::stride, :].reshape(-1, 1)
        true_output = np.squeeze(true,axis=-1)[::stride, :].reshape(-1, 1)

        pred_res = pd.DataFrame({'pred': pred_output.flatten(), 'true': true_output.flatten()})
        pred_res.loc[pred_res['true'] < 1e-3, 'true'] = 0
        pred_res.loc[pred_res['true'] < 1e-3, 'pred'] = 0

        [mse, rmse,nrmse, mae,mape,rae, r2,corr] = results_evaluation(pred_res['true'].values, pred_res['pred'].values)

        pred_res.to_csv(os.path.join(path, 'pred_results_{}_{}.csv'.format(self.args.data,self.args.data_path[:-4])))
        metrics_df = pd.DataFrame({'trainable_params':trainable_params,'mse':mse,'rmse': rmse,'nrmse':nrmse, 'mae': mae, 'mape': mape,'rae':rae,'r2': r2,'corr':corr}, index=[0])
        metrics_df.to_csv(os.path.join(path, 'metrics_results_{}_{}.csv'.format(self.args.data,self.args.data_path[:-4])))

        print('RMSE: {} MAE: {} R2: {}'.format(rmse, mae, r2))
        return pred_res,metrics_df

    def res_evaluation_multi_target(self,true,pred,trainable_params,path):
        stride = self.args.pred_len
        true = true[::stride,:,:].reshape(-1,len(self.args.target))
        pred = pred[::stride,:,:].reshape(-1,len(self.args.target))
        columns_list = []
        for i in self.args.target:
            columns_list.append('{}_true'.format(i))
            columns_list.append('{}_pred'.format(i))
        res_df = pd.DataFrame(columns=columns_list)
        mse_list, rmse_list, mae_list, r2_list, corr_list,mape_list = [], [], [], [], [], []
        nrmse_list,rae_list = [],[]
        for i in self.args.target:
            res_df['{}_pred'.format(i)] = pred[:, self.args.target.index(i)]
            res_df['{}_true'.format(i)] = true[:, self.args.target.index(i)]
            self._show_plot(i,y_true=true[:, self.args.target.index(i)],y_pred=pred[:, self.args.target.index(i)],path=path)

            [mse, rmse,nrmse, mae,mape,rae, r2,corr] = results_evaluation(true[:, self.args.target.index(i)], pred[:, self.args.target.index(i)])
            print('{} mse:{}, rmse:{} mae:{} r2:{} corr:{}'.format(i, mse, rmse, mae, r2, corr))
            np.save(os.path.join(path, 'metrics_{}.npy'.format(i)), np.array([mse, rmse, mae, r2, corr]))

            mse_list.append(mse)
            rmse_list.append(rmse)
            nrmse_list.append(nrmse)
            mae_list.append(mae)
            mape_list.append(mape)
            rae_list.append(rae)
            r2_list.append(r2)
            corr_list.append(corr)

        res_metrics_df = pd.DataFrame(columns=['trainable_params','mse', 'rmse','nrmse', 'mae','mape','rae', 'r2','corr'],
                                      index=[i for i in self.args.target])
        res_metrics_df['trainable_params'] = trainable_params
        res_metrics_df['mse'] = mse_list
        res_metrics_df['rmse'] = rmse_list
        res_metrics_df['nrmse'] = nrmse_list
        res_metrics_df['rae'] = rae_list
        res_metrics_df['mae'] = mae_list
        res_metrics_df['mape'] = mape_list
        res_metrics_df['r2'] = r2_list
        res_metrics_df['corr'] = corr_list
        res_metrics_df.loc['mean'] = res_metrics_df.mean()
        print(res_metrics_df.loc['mean'])
        res_df.to_csv(os.path.join(path, 'pred_res_{}.csv'.format(self.args.data_path[:-4])))
        res_metrics_df.to_csv(os.path.join(path, 'res_metrics_df_{}.csv'.format(self.args.data_path[:-4])))
        return res_df,res_metrics_df

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
