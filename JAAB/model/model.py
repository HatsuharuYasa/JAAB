"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import random
import torch.nn.functional as F
import math


#Local imports
from model.modules import BaseRGBModel, EDSGPMIXERLayers, FCLayers, FC2Layers, CrossAttentionLayer, CoordEncoding, step, process_prediction, process_double_head, process_labels
from model.shift import make_temporal_shift

class TDEEDModel(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._modality = args.modality
            assert self._modality == 'rgb', 'Only RGB supported for now'
            in_channels = {'rgb': 3}[self._modality]
            self._temp_arch = args.temporal_arch
            assert self._temp_arch in ['ed_sgp_mixer'], 'Only ed_sgp_mixer supported for now'
            self._radi_displacement = args.radi_displacement
            self._feature_arch = args.feature_arch
            assert 'rny' in self._feature_arch, 'Only rny supported for now'
            self._double_head = False

            if self._feature_arch.startswith(('rny002', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args._feature_arch)

            # Add Temporal Shift Modules
            self._require_clip_len = -1
            if self._feature_arch.endswith('_gsm'):
                make_temporal_shift(features, args.clip_len, mode='gsm')
                self._require_clip_len = args.clip_len
            elif self._feature_arch.endswith('_gsf'):
                make_temporal_shift(features, args.clip_len, mode='gsf')
                self._require_clip_len = args.clip_len

            self._features = features
            self._feat_dim = self._d
            feat_dim = self._d

            #Cross attention layer
            self._cross_att = CrossAttentionLayer(
                embed_dim=self._d,
                num_heads=8,
                dropout=0.1,
            )

            #Temporal encoding
            self.temp_enc = nn.Parameter(torch.normal(mean = 0, std = 1 / args.clip_len, size = (args.clip_len, self._d)))

            #Coordinate encoding
            self._pos_enc = CoordEncoding(
                grid_size_x=7,
                grid_size_y=7,
                embed_dim=self._d,
            )
            
            if self._temp_arch == 'ed_sgp_mixer':
                self._temp_fine = EDSGPMIXERLayers(feat_dim, args.clip_len, num_layers=args.n_layers, ks = args.sgp_ks, k = args.sgp_r, concat = True)
                self._pred_fine = FCLayers(self._feat_dim, args.num_classes+1)
            else:
                raise NotImplementedError(self._temp_arch)
            
            if self._radi_displacement > 0:
                self._pred_displ = FCLayers(self._feat_dim, 1)
            
            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

            #Augmentation at test time
            self.augmentationI = T.Compose([
                T.RandomHorizontalFlip(p = 1.0)
            ])

            #Croping in case of using it
            self.croping = args.crop_dim
            if self.croping != None:
                self.cropT = T.RandomCrop((self.croping, self.croping))
                self.cropI = T.CenterCrop((self.croping, self.croping))
            else:
                self.cropT = torch.nn.Identity()
                self.cropI = torch.nn.Identity()

        def forward(self, x_c, x_f, pos, y = None, inference=False, augment_inference=False):
            
            x_c, x_f = self.normalize(x_c), self.normalize(x_f) #Normalize to 0-1
            
            # Acquire the shape of each input
            batch_size_c, true_clip_len_c, channels_c, height_c, width_c = x_c.shape
            batch_size_f, clip_len_f, channels_f, height_f, width_f = x_f.shape

            assert batch_size_c == batch_size_f, 'Batch size of local and global views must be the same'
            
            ## Process the local view ##

            if not inference:
                #x_c.view(-1, channels_c, height_c, width_c)
                #if self.croping != None:
                #    height_c = self.croping
                #    width_c = self.croping
                #x_c = self.cropT(x_c) #same crop for all frames
                #x_c = x_c.view(batch_size_c, true_clip_len_c, channels_c, height_c, width_c)
                x_c = self.augment(x_c) #augmentation per-batch
                x_c = self.standarize(x_c) #standarization imagenet stats

            else:
                #x_c = x_c.view(-1, channels_c, height, width)
                #if self.croping != None:
                #    height = self.croping
                #    width = self.croping
                #x_c = self.cropI(x_c) #same center crop for all frames
                #x_c = x_c.view(batch_size_c, true_clip_len_c, channels_c, height_c, width_c)
                if augment_inference:
                    x_c = self.augmentI(x_c)
                x_c = self.standarize(x_c)

            clip_len_c = true_clip_len_c

            ## Process the global view ##

            if not inference:
                x_f.view(-1, channels_f, height_f, width_f)
                if self.croping != None:
                    height_f = self.croping
                    width_f = self.croping
                x_f = self.cropT(x_f)
                x_f = x_f.view(batch_size_f, clip_len_f, channels_f, height_f, width_f)
                x_f = self.augment(x_f)
                x_f = self.standarize(x_f)
            else:
                x_f = x_f.view(-1, channels_f, height_f, width_f)
                if self.croping != None:
                    height_f = self.croping
                    width_f = self.croping
                x_f = self.cropI(x_f)
                x_f = x_f.view(batch_size_f, clip_len_f, channels_f, height_f, width_f)
                if augment_inference:
                    x_f = self.augmentI(x_f)
                x_f = self.standarize(x_f)
            
            clip_len_f = clip_len_f

            ## Process the coordinate encoding ##
            _pos = pos.view(-1, 2)
            # WARNING THIS IS A HARCODED, DONT FORGET TO CHANGE IT
            _pos[:, 0], _pos[:, 1] = _pos[:, 0] / 1920, _pos[:, 1] / 1080 # Normalize the coordinates
            _pos_enc = self._pos_enc(_pos[:, 0], _pos[:, 1])
            _pos_enc = _pos_enc.view(batch_size_c, true_clip_len_c, -1)
                        
            # Process local and global features separately
            #assert torch.all(torch.isfinite(x_c))
            im_feat_c = self._features(
                x_c.view(-1, channels_c, height_c, width_c)
            ).reshape(batch_size_c, clip_len_c, self._d)

            im_feat_f = self._features(
                x_f.view(-1, channels_f, height_f, width_f)
            ).reshape(batch_size_f, clip_len_f, self._d)

            # Forward to cross attention
            im_feat = self._cross_att(im_feat_c, im_feat_f, _pos_enc)

            im_feat = im_feat + self.temp_enc.expand(batch_size_c, -1, -1)

            if self._temp_arch == 'ed_sgp_mixer':
                im_feat = self._temp_fine(im_feat)
                if self._radi_displacement > 0:
                    displ_feat = self._pred_displ(im_feat).squeeze(-1)
                    im_feat = self._pred_fine(im_feat)
                    return {'im_feat': im_feat, 'displ_feat': displ_feat}, y
                im_feat = self._pred_fine(im_feat)
                return im_feat, y
            
            else:
                raise NotImplementedError(self._temp_arch)
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x
        
        def augmentI(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentationI(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x
        
        def update_pred_head(self, num_classes = [1, 1]):
            self._pred_fine = FC2Layers(self._feat_dim, num_classes)
            self._pred_fine = self._pred_fine.cuda()
            self._double_head = True

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))
            print('  CNN features:',
                sum(p.numel() for p in self._features.parameters()))
            print('  Coord Encoding:',
                sum(p.numel() for p in self._pos_enc.parameters()))
            print('  Cross Attention:',
                sum(p.numel() for p in self._cross_att.parameters()))
            print('  Temporal:',
                sum(p.numel() for p in self._temp_fine.parameters()))
            print('  Head:',
                sum(p.numel() for p in self._pred_fine.parameters()))

    def __init__(self, device='cuda', args=None):
        self.device = device
        self._model = TDEEDModel.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(device)
        self._num_classes = args.num_classes + 1

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None,
            acc_grad_iter=1, fg_weight=2, valMAP=False):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        if valMAP:
            map_labels = []
            map_preds = []

        ce_kwargs = {}
        #Defining custom weight for focus task
        custom_fg_weight = [1, fg_weight, fg_weight, fg_weight * 2]
        ce_kwargs['weight'] = torch.FloatTensor(custom_fg_weight).to(self.device)

        #if fg_weight != 1:
        #    ce_kwargs['weight'] = torch.FloatTensor(
        #        [1] + [fg_weight] * (self._num_classes - 1)).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame_close = batch['frame_close'].to(self.device).float()
                frame_wide = batch['frame_wide'].to(self.device).float()
                track = batch['track'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device)

                #update labels for double head
                if self._model._double_head:
                    batch_dataset = batch['dataset']
                    label = update_labels_2heads(label, batch_dataset, self._args.num_classes)

                if 'labelD' in batch.keys():
                    labelD = batch['labelD'].to(self.device).float()
                
                if 'frame_close2' in batch.keys():
                    frame_close2 = batch['frame_close2'].to(self.device).float()
                    frame_wide2 = batch['frame_wide2'].to(self.device).float()
                    track2 = batch['track2'].to(self.device).float()
                    label2 = batch['label2']
                    label2 = label2.to(self.device)

                    if 'labelD2' in batch.keys():
                        labelD2 = batch['labelD2'].to(self.device).float()
                        labelD_dist = torch.zeros((labelD.shape[0], label.shape[1])).to(self.device)

                    l = [random.betavariate(0.2, 0.2) for _ in range(frame_close2.shape[0])]

                    label_dist = torch.zeros((label.shape[0], label.shape[1], self._num_classes)).to(self.device)

                    for i in range(frame_close2.shape[0]):
                        frame_close[i] = l[i] * frame_close[i] + (1 - l[i]) * frame_close2[i]
                        frame_wide[i] = l[i] * frame_wide[i] + (1 - l[i]) * frame_wide2[i]
                        track[i] = l[i] * track[i] + (1 - l[i]) * track2[i]
                        lbl1 = label[i]
                        lbl2 = label2[i]

                        label_dist[i, range(label.shape[1]), lbl1] += l[i]
                        label_dist[i, range(label2.shape[1]), lbl2] += 1 - l[i]

                        if 'labelD2' in batch.keys():
                            labelD_dist[i] = l[i] * labelD[i] + (1 - l[i]) * labelD2[i]

                    label = label_dist
                    if 'labelD2' in batch.keys():
                        labelD = labelD_dist

                if valMAP:
                    labels_aux = process_labels(label, labelD if 'labelD' in batch.keys() else None,
                                        num_classes = self._num_classes)
                    map_labels.append(labels_aux.cpu())

                # Depends on whether mixup is used
                label = label.flatten() if len(label.shape) == 2 \
                    else label.view(-1, label.shape[-1])

                with torch.amp.autocast('cuda'):
                    pred, y = self._model(frame_close, frame_wide, track, y = label, inference=inference)

                    if 'labelD' in batch.keys():
                        predD = pred['displ_feat']
                        pred = pred['im_feat']

                    if valMAP:
                        pred_aux = process_prediction(pred, predD)
                        map_preds.append(pred_aux.cpu())

                    loss = 0.

                    if self._model._double_head:
                        b, t, c = pred.shape
                        if len(label.shape) == 2:
                            label = label.view(b, t, c)
                        if len(label.shape) == 1:
                            label = label.view(b, t)

                        for i in range(pred.shape[0]):
                            if batch_dataset[i] == 1:
                                if len(label.shape) == 3:
                                    aux_label = label[i][:, :self._args.num_classes+1]
                                elif len(label.shape) == 2:
                                    aux_label = label[i]
                                else:
                                    raise NotImplementedError
                                    
                                loss += F.cross_entropy(pred[i][:, :self._args.num_classes+1], aux_label,
                                                        weight = ce_kwargs['weight'][:self._args.num_classes+1]) / (pred.shape[0])
                                    
                            elif batch_dataset[i] == 2:
                                if len(label.shape) == 3:
                                    aux_label = label[i][:, self._args.num_classes+1:]
                                elif len(label.shape) == 2:
                                    aux_label = label[i] - (self._args.num_classes + 1)
                                else:
                                    raise NotImplementedError
                                    
                                loss += F.cross_entropy(pred[i][:, self._args.num_classes+1:], aux_label,
                                                        weight = ce_kwargs['weight'][:self._args.pretrain['num_classes']+1]) / (pred.shape[0])

                    else:
                        predictions = pred.reshape(-1, self._num_classes)

                        #print("DEBUG: {} \t {}".format(predictions.shape, label.shape))
                        loss += F.cross_entropy(
                            predictions, label,
                            **ce_kwargs)    
                    
                            
                    if 'labelD' in batch.keys():
                        lossD = F.mse_loss(predD, labelD, reduction = 'none')
                        lossD = (lossD).mean()
                        loss = loss + lossD

                if optimizer is not None:
                    step(optimizer, scaler, loss / acc_grad_iter,
                        lr_scheduler=lr_scheduler,
                        backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                epoch_loss += loss.detach().item()
                #if 'labelD' in batch.keys():
                #    epoch_lossD += lossD.detach().item()
        
        if valMAP:
            return epoch_loss / len(loader), torch.cat(map_labels, 0), torch.cat(map_preds, 0)
        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq_c, seq_f, track, use_amp=True, augment_inference = False):
        
        if not isinstance(seq_c, torch.Tensor):
            seq_c = torch.FloatTensor(seq_c)
        if not isinstance(seq_f, torch.Tensor):
            seq_f = torch.FloatTensor(seq_f)

        if len(seq_c.shape) == 4: # (L, C, H, W)
            seq_c = seq_c.unsqueeze(0)
        if len(seq_f.shape) == 4: # (L, C, H, W)
            seq_f = seq_f.unsqueeze(0)
        if len(track.shape) == 2: # (L, XY)
            track = track.unsqueeze(0)

        if seq_c.device != self.device:
            seq_c = seq_c.to(self.device)
        if seq_f.device != self.device:
            seq_f = seq_f.to(self.device)
        if track.device != self.device:
            track = track.to(self.device)
        
        seq_c = seq_c.float()
        seq_f = seq_f.float()

        self._model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda') if use_amp else nullcontext():
                pred, y = self._model(seq_c, seq_f, track, inference=True, augment_inference = augment_inference)
            if isinstance(pred, dict):
                predD = pred['displ_feat']
                pred = pred['im_feat']
                if isinstance(pred, list):
                    pred = pred[0]
                if isinstance(predD, list):
                    predD = predD[0]
                if self._model._double_head:
                    pred = process_double_head(pred, predD, num_classes = self._args.num_classes+1)
                else:
                    pred = process_prediction(pred, predD)
                pred_cls = torch.argmax(pred, axis=2)
                return pred_cls.cpu().numpy(), pred.cpu().numpy()
            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            else:
                pred = torch.softmax(pred, axis=2)

            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()
        
def update_labels_2heads(labels, datasets, num_classes1 = 1):
    for i in range(len(datasets)):
        if datasets[i] == 2:
            labels[i] = labels[i] + num_classes1 + 1

    return labels
