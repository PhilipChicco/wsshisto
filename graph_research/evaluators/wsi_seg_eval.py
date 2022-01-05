import torch, os, copy, cv2
import torch.optim as optim
import numpy as np

from pycm import *
from tqdm import tqdm
from torchvision import transforms, utils
import torch.nn.functional as F
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from evaluators.base_eval import BaseTester
import trainers.wsi_seam_utils as visualization

from loaders import get_wsicompressed_test
from utils.misc import AverageMeter, get_metrics
from utils.post_seg import Watershed, get_scores, overlap_preds
from histomicstk.saliency.tissue_detection import get_tissue_mask
from loaders.utils_wsi import HistoNormalize

class WSISegTest(BaseTester):

    def __init__(self, cfg, writer, logger):
        super().__init__(cfg, writer, logger)

        self.norm = HistoNormalize()

        # setup datasets
        print('Setting up data ...')
        self.data_transforms = {
            'test': transforms.Compose([transforms.ToTensor()]),
        }

        loaders_dict = get_wsicompressed_test(cfg, self.data_transforms,
                                              use_json=True,
                                              mask_dir=self.cfg['data']['cam_path'])
        self.test_dset, self.test_loader = loaders_dict['test']
        
        print()


    def _run_test(self):
        thr_values = list(self.cfg['testing']['threshold_list'].split(","))
        dice_score, iou_score, acc_score, score_list = self.inf_seg(thr_values)
        fp2 = open(os.path.join(self.logdir, 'all_scores.txt'), 'w')
        for th in thr_values:
            fp = open(os.path.join(self.logdir, f'predictions_seg_{th}.csv'), 'w')
            fp.write('file,dice,iou,acc\n')
            for item in score_list[str(th)]:
                fp.write('{},{:.4f},{:.4f},{:.4f}\n'.format(item[0], item[1],item[2], item[3]))
            fp.write('Average,{:.4f},{:.4f},{:.4f}\n'.format(dice_score[str(th)].avg(),iou_score[str(th)].avg(),acc_score[str(th)].avg()))
            fp.close()
            print('--- (test) | THRESHOLD: {} | DICE {:.4f} | mIOU {:.4f} | pACC {:.4f} '.format(th,
            dice_score[str(th)].avg(),iou_score[str(th)].avg(),acc_score[str(th)].avg()))
            fp2.write('--- (test) | THRESHOLD: {} | DICE {:.4f} | mIOU {:.4f} | pACC {:.4f}\n'.format(th,
            dice_score[str(th)].avg(),iou_score[str(th)].avg(),acc_score[str(th)].avg()))
        fp2.close()

    def inf_seg(self, thr_values):

        dice_scores = {str(i):AverageMeter() for i in thr_values }
        iou_scores  = {str(i):AverageMeter() for i in thr_values }
        acc_scores  = {str(i):AverageMeter() for i in thr_values }
        score_list  = {str(i):[] for i in thr_values }
        thr = self.cfg['testing']['threshold']

        self.model.eval()
        os.makedirs(os.path.join(self.logdir, '0'), exist_ok=True)
        os.makedirs(os.path.join(self.logdir, '1'), exist_ok=True)

        my_cm = matplotlib.cm.get_cmap('jet')

        with torch.no_grad():
            final_itr = tqdm(self.test_loader, ncols=80, desc='Inference (Segmentation)...')
            #
            for i, (input, targets, mask_label ,slide_name) in enumerate(final_itr):
                target = targets.data.cpu().numpy()[0]
                
                if target == 0: continue

                input      = input.to(self.device)
                slide_name = slide_name[0]
                mask_label = mask_label.data.cpu().numpy()[0]


                mask_path  = os.path.join(self.cfg['testing']['masks'], slide_name + '_tissue_fig.png')
                input_i    = Image.open(mask_path).convert('RGB')
                input_i_t  = np.array(self.norm(input_i))
                mask_size  = input_i.size
                mask_size  = (mask_size[1],mask_size[0])
                input_i_np = np.array(input_i)

                # generate tissue mask with deconv
                input_i_t[input_i_t.copy() == 0] = 255
                input_i_t = np.uint8(input_i_t)

                # generate tissue mask with deconv
                mask_tissue = get_tissue_mask(
                    input_i_t, deconvolve_first=True, # 1 thresholds : for COLON/AMC
                    n_thresholding_steps=1, sigma=0., min_size=1)[0]
                mask = np.clip(mask_tissue,0,1)

                #

                mask_label = cv2.resize(mask_label.astype(np.float32), mask_size, interpolation=cv2.INTER_NEAREST)
                mask_label = np.transpose(mask_label, [1, 0])
                mask_label[mask == 0] = 0


                img = input_i_np.astype(np.float32) / 255.0
                img[:, :, 0][mask_label > 0] = 0
                img[:, :, 1][mask_label > 0] = 5
                img[:, :, 2][mask_label > 0] = 0
                input_mask = np.array(Image.fromarray(np.uint8(img * 255)))
                #

                cam   = self.model(input)
                cam   = F.softmax(cam,1)
                cam   = F.interpolate(cam, size=(mask_size[1],mask_size[0]), mode='bilinear', align_corners=True).squeeze(1)[0]
                cam_n = np.transpose(cam.data.cpu().numpy(), [0, 2, 1])
               

                # save preds 
                #cam_save = copy.deepcopy(cam_n[1,:,:] )
                #save_pth = os.path.join(self.logdir, str(target), '{}.npy'.format(slide_name))
                #np.save(save_pth, cam_save)


                ####
                for th_val in thr_values:
                    cam_pred = copy.deepcopy(cam_n[1,:,:] ) 
                    cam_pred[mask == 0] = 0.0
                    cam_pred[cam_pred < float(th_val) ] = 0.0
                    cam_pred[cam_pred >= float(th_val)] = 1.0 
                    pred_ws = np.clip(cam_pred.copy(), 0, 1).astype(np.int)

                    if target == 1:
                        if np.sum(cam_pred) > 0:  # avoid zero division
                            dice, iou, acc = get_scores(mask_label.astype(np.int), cam_pred)
                        else:
                            dice, iou, acc = 0.0, 0.0, 0.0
                        
                        dice_scores[str(th_val)].append(dice)
                        iou_scores[str(th_val)].append(iou)
                        acc_scores[str(th_val)].append(acc)
                        score_list[str(th_val)].append((slide_name,dice,iou,acc))
                

                #####
                #selected threshold
                cam_pred = copy.deepcopy(cam_n[1,:,:])  
                cam_pred[mask == 0] = 0.0
                cam_pred[cam_pred < thr ] = 0.0
                cam_pred[cam_pred >= thr] = 1.0 
                pred_ws = np.clip(cam_pred.copy(), 0, 1).astype(np.int)

                img = input_i_np.astype(np.float32)/255.0
                img[:, :, 0][pred_ws > 0] = 5
                img[:, :, 1][pred_ws > 0] = 0
                img[:, :, 2][pred_ws > 0] = 0
                cam_blend_np = np.array(Image.fromarray(np.uint8(img * 255)))

                ####save for tumors
                plt.figure()
                cam_n = copy.deepcopy(cam_n[1,:,:])
                cam_n[mask == 0]  = 0.0
                cam_n = my_cm(cam_n)
                cam_n = np.uint8(255 * cam_n)
                cam_n = Image.fromarray(cam_n).convert('RGB')
                cam_n_np = np.array(cam_n)

                inp_i = torch.from_numpy(input_i_np.transpose(2, 0, 1))
                inp_m = torch.from_numpy(input_mask.transpose(2, 0, 1))
                cam_j = torch.from_numpy(cam_n_np.transpose(2, 0, 1))
                cam_o = torch.from_numpy(cam_blend_np.transpose(2, 0, 1))

                input_grid   = utils.make_grid(inp_i.float(), nrow=1).unsqueeze_(0)
                input_mask   = utils.make_grid(inp_m.float(), nrow=1).unsqueeze_(0)
                cam_jts_grid = utils.make_grid(cam_j.float(), nrow=1).unsqueeze_(0)
                cam_ovr_grid = utils.make_grid(cam_o.float(), nrow=1).unsqueeze_(0)

                ## save individual cams
                # INPUT | GT | HEATMAP | SEG
                save_pth = os.path.join(self.logdir, str(target), '{}_grid.png'.format(slide_name))
                cat_i = torch.cat([input_grid, input_mask, cam_ovr_grid,cam_jts_grid], 0)
                utils.save_image(cat_i, fp=save_pth,  scale_each=True, normalize=True)


        del my_cm
        return dice_scores,iou_scores,acc_scores, score_list

    def _on_test_end(self):
        pass

    def _on_test_start(self):
        pass