import torch

from collections import OrderedDict

from utils import misc
from models.base_models import BaseModel
from models.modules.base_modules import ModuleFactory


class ModelCombine(BaseModel):

    def __init__(self, opt):

        super(ModelCombine, self).__init__(opt)

        self._opt = opt

        self._init_create_networks()

        if self._is_train:
            self._init_train_vars()

        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        self._bs = self._opt.batch_size

        # init input
        self._input_mask = self._Tensor([0])
        self._input_img_LR_masked = self._Tensor([0])
        self._input_img_LR = self._Tensor([0])
        self._input_img_SR = self._Tensor([0])
        self._input_img_landmark = self._Tensor([0])
        self._input_img_face_parsing = self._Tensor([0])
        self._input_point = self._Tensor([0])

        # init visual
        self._vis_batch_mask = self._Tensor([0])
        self._vis_batch_img_LR_masked = self._Tensor([0])
        self._vis_batch_img_LR_fc = self._Tensor([0])
        self._vis_batch_img_LR_syn = self._Tensor([0])
        self._vis_batch_img_LR = self._Tensor([0])

        self._vis_batch_img_coarse = self._Tensor([0])
        self._vis_batch_img_fine = self._Tensor([0])
        self._vis_batch_img_patch = self._Tensor([0])
        self._vis_batch_img_SR = self._Tensor([0])

        self._vis_batch_img_landmark = self._Tensor([0])
        self._vis_batch_img_landmark_GT = self._Tensor([0])

        self._vis_batch_img_face_parsing = self._Tensor([0])
        self._vis_batch_img_face_parsing_GT = self._Tensor([0])

        # init loss
        self._loss_1_hole = self._Tensor([0])
        self._loss_1_vaild = self._Tensor([0])
        self._loss_1_sty = self._Tensor([0])
        self._loss_1_per = self._Tensor([0])
        self._loss_1_synth_smooth = self._Tensor([0])

        self._loss_2_coarse = self._Tensor([0])
        self._loss_2_landmark = self._Tensor([0])
        self._loss_2_face_parsing = self._Tensor([0])

        self._loss_2_fine = self._Tensor([0])
        self._loss_2_per = self._Tensor([0])

        self._loss_g_global_adv = self._Tensor([0])

        self._loss_d_global_adv = self._Tensor([0])
        self._loss_d_global_adv_gp = self._Tensor([0])

    def _init_create_networks(self):

        if self._opt.fc_module == 'pconv':
            self._FC = ModuleFactory.get_by_name(self._opt.fc_module, self._opt.freeze_enc_bn).to(self._device)
        else:
            raise ValueError('error!')

        if self._opt.fc_pretrain:

            self._FC.load_state_dict(torch.load(self._opt.fc_pretrain_model_path))
            self._FC.eval()

        self._SR = ModuleFactory.get_by_name(self._opt.sr_module).to(self._device)
        self._vgg = ModuleFactory.get_by_name('vgg').to(self._device)

        self._global_d = ModuleFactory.get_by_name('d').to(self._device)

    def _init_train_vars(self):

        self._current_lr_g = self._opt.lr_g
        if self._opt.fix_fc:
            self._opti_g = torch.optim.Adam(self._SR.parameters(), lr=self._current_lr_g,
                                            betas=[self._opt.g_adam_b1, self._opt.g_adam_b2])
        else:
            self._opti_g = torch.optim.Adam(list(self._FC.parameters()) + list(self._SR.parameters()),
                                            lr=self._current_lr_g, betas=[self._opt.g_adam_b1, self._opt.g_adam_b2])

        self._current_lr_d = self._opt.lr_d
        self._opti_global_d = torch.optim.Adam(self._global_d.parameters(), lr=self._current_lr_d,
                                               betas=[self._opt.d_adam_b1, self._opt.d_adam_b2])

    def set_input(self, input_dict):

        self._input_mask = input_dict['mask'].to(self._device)
        self._input_img_LR_masked = input_dict['img_LR_masked'].to(self._device)
        self._input_img_LR = input_dict['img_LR'].to(self._device)
        self._input_img_SR = input_dict['img_SR'].to(self._device)
        self._input_img_landmark = input_dict['img_landmark'].to(self._device)
        self._input_img_face_parsing = input_dict['img_face_parsing'].to(self._device)
        self._input_point = input_dict['point'].to(self._device)

    def set_train(self):

        if self._opt.fix_fc:
            self._SR.train()
        else:
            self._FC.train()
            self._SR.train()

        self._global_d.train()
        self._is_train = True

    def set_eval(self):

        if self._opt.fix_fc:
            self._SR.eval()
        else:
            self._FC.eval()
            self._SR.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False):

        # if not self._is_train:
        # print('........')
        if self._opt.fc_module == 'pconv':
            img_fc, _ = self._FC.forward(self._input_img_LR_masked, self._input_mask)
        elif self._opt.fc_module == 'gfc_128' or self._opt.fc_module == 'gfc_32':
            img_fc = self._FC.forward(self._input_img_LR_masked)
        else:
            raise ValueError('error')

        img_synth = img_fc * (1 - self._input_mask) + self._input_mask * self._input_img_LR
        img_coarse_sr, img_coarse_sr_landmark, img_coarse_sr_fp, img_fine_sr = self._SR.forward(img_synth)

        if keep_data_for_visuals:

            self._vis_batch_mask = misc.tensor2im(self._input_mask[:self._opt.show_max], idx=-1, nrows=1)
            self._vis_batch_img_LR_masked = misc.tensor2im(
                self._input_img_LR_masked[:self._opt.show_max], idx=-1, nrows=1)
            self._vis_batch_img_LR_fc = misc.tensor2im(img_fc[:self._opt.show_max], idx=-1, nrows=1)
            self._vis_batch_img_LR_syn = misc.tensor2im(img_synth[:self._opt.show_max], idx=-1, nrows=1)
            self._vis_batch_img_LR = misc.tensor2im(self._input_img_LR[:self._opt.show_max], idx=-1, nrows=1)

            self._vis_batch_img_coarse = misc.tensor2im(img_coarse_sr.data[:self._opt.show_max], idx=-1, nrows=1)
            self._vis_batch_img_fine = misc.tensor2im(img_fine_sr.data[:self._opt.show_max], idx=-1, nrows=1)
            self._vis_batch_img_SR = misc.tensor2im(self._input_img_SR[:self._opt.show_max], idx=-1, nrows=1)

            self._vis_batch_img_landmark = misc.landmark2im(img_coarse_sr_landmark.data[:self._opt.show_max])
            self._vis_batch_img_landmark_GT = misc.landmark2im(self._input_img_landmark.data[:self._opt.show_max])

            face_parsing_v = torch.argmax(img_coarse_sr_fp, dim=1, keepdim=False)
            self._vis_batch_img_face_parsing = misc.faceparsing2im(face_parsing_v.data[:self._opt.show_max])

            # self._vis_batch_img_face_parsing_GT = misc.faceparsing2im(
            #     self._input_img_face_parsing.squeeze_().unsqueeze_(0).data[:self._opt.show_max])
            self._vis_batch_img_face_parsing_GT = misc.faceparsing2im(
                self._input_img_face_parsing.squeeze_().data[:self._opt.show_max])


    def optimize_parameters(self, train_generator, keep_data_for_visuals=False):

        if self._is_train:

            loss_d_global_adv, synth_img_global = self._forward_global_d()
            loss_d_global_adv = loss_d_global_adv * self._opt.lambda_global_d_prob
            self._opti_global_d.zero_grad()
            loss_d_global_adv.backward()
            self._opti_global_d.step()
            self._loss_d_global_adv = loss_d_global_adv

            loss_d_global_adv_gp = self._gradinet_penalty_d(synth_img_global, self._input_img_SR,
                                                            self._global_d) * self._opt.lambda_global_d_gp
            self._opti_global_d.zero_grad()
            loss_d_global_adv_gp.backward()
            self._opti_global_d.step()
            self._loss_d_global_adv_gp = loss_d_global_adv_gp

            if train_generator:

                loss_g = self._forward_g(keep_data_for_visuals)

                self._opti_g.zero_grad()
                loss_g.backward()
                self._opti_g.step()

    def _forward_g(self, keep_data_for_visuals):

        if self._opt.fc_module == 'pconv':
            img_fc, _ = self._FC.forward(self._input_img_LR_masked, self._input_mask)
        elif self._opt.fc_module == 'gfc_128' or self._opt.fc_module == 'gfc_32':
            img_fc = self._FC.forward(self._input_img_LR_masked)
        else:
            raise ValueError('error')

        img_synth = img_fc * (1 - self._input_mask) + self._input_mask * self._input_img_LR

        img_coarse_sr, img_coarse_sr_landmark, img_coarse_sr_fp, img_fine_sr = self._SR.forward(img_synth)

        if self._opt.fix_fc is False:

            loss_1_hole, loss_1_vaild, loss_1_sty, loss_1_per, loss_1_synth_smooth = \
                self._inpainting_loss(img_synth, img_fc, self._input_img_LR)

            self._loss_1_hole = loss_1_hole * self._opt.lambda_loss_1_hole
            self._loss_1_vaild = loss_1_vaild * self._opt.lambda_loss_1_vaild
            self._loss_1_sty = loss_1_sty * self._opt.lambda_loss_1_sty
            self._loss_1_per = loss_1_per * self._opt.lambda_loss_1_per
            self._loss_1_synth_smooth = loss_1_synth_smooth * self._opt.lambda_loss_1_synth_smooth

        self._loss_2_coarse = misc.compute_loss_l2(img_coarse_sr, self._input_img_SR) * self._opt.lambda_loss_2_coarse
        self._loss_2_landmark = misc.compute_loss_l2(img_coarse_sr_landmark,
                                                     self._input_img_landmark) * self._opt.lambda_loss_2_landmark
        self._loss_2_face_parsing = misc.compute_loss_cross_entropy(
            img_coarse_sr_fp, self._input_img_face_parsing) * self._opt.lambda_loss_2_parsing

        img_fine_sr_feature = self._vgg(img_fine_sr)
        img_gt_feature = self._vgg(self._input_img_SR)
        loss_per = 0
        for i in range(len(img_fine_sr_feature)):
            loss_per += misc.compute_loss_l1(img_fine_sr_feature[i], img_gt_feature[i])

        self._loss_2_per = loss_per * self._opt.lambda_loss_2_per

        self._loss_2_fine = misc.compute_loss_l2(img_fine_sr, self._input_img_SR) * self._opt.lambda_loss_2_fine

        d_fake_global_prob = self._global_d.forward(img_fine_sr)
        self._loss_g_global_adv = misc.compute_loss_d(d_fake_global_prob, True) * self._opt.lambda_global_d_prob

        if keep_data_for_visuals:

            self._vis_batch_mask = misc.tensor2mask(self._input_mask[:self._opt.show_max], idx=-1, nrows=1)
            self._vis_batch_img_LR_masked = misc.tensor2im(
                self._input_img_LR_masked[:self._opt.show_max], idx=-1, nrows=1)
            self._vis_batch_img_LR_fc = misc.tensor2im(img_fc[:self._opt.show_max], idx=-1, nrows=1)
            self._vis_batch_img_LR_syn = misc.tensor2im(img_synth[:self._opt.show_max], idx=-1, nrows=1)
            self._vis_batch_img_LR = misc.tensor2im(self._input_img_LR[:self._opt.show_max], idx=-1, nrows=1)

            self._vis_batch_img_coarse = misc.tensor2im(img_coarse_sr.data[:self._opt.show_max], idx=-1, nrows=1)
            self._vis_batch_img_fine = misc.tensor2im(img_fine_sr.data[:self._opt.show_max], idx=-1, nrows=1)

            self._vis_batch_img_SR = misc.tensor2im(self._input_img_SR[:self._opt.show_max], idx=-1, nrows=1)

            self._vis_batch_img_landmark = misc.landmark2im(img_coarse_sr_landmark.data[:self._opt.show_max])
            self._vis_batch_img_landmark_GT = misc.landmark2im(self._input_img_landmark.data[:self._opt.show_max])

            face_parsing_v = torch.argmax(img_coarse_sr_fp, dim=1, keepdim=False)
            self._vis_batch_img_face_parsing = misc.faceparsing2im(face_parsing_v.data[:self._opt.show_max])
            self._vis_batch_img_face_parsing_GT = misc.faceparsing2im(
                self._input_img_face_parsing.squeeze_().data[:self._opt.show_max])

        loss_1 = self._loss_1_hole + self._loss_1_vaild + self._loss_1_sty + \
            self._loss_1_per + self._loss_1_synth_smooth

        loss_2 = self._loss_2_per + self._loss_2_fine + self._loss_2_coarse + \
            self._loss_2_landmark + self._loss_2_face_parsing

        loss_g_d = self._loss_g_global_adv

        return loss_1 + loss_2 + loss_g_d

    def _forward_global_d(self):

        if self._opt.fc_module == 'pconv':
            img_fc, _ = self._FC.forward(self._input_img_LR_masked, self._input_mask)
        elif self._opt.fc_module == 'gfc_128' or self._opt.fc_module == 'gfc_32':
            img_fc = self._FC.forward(self._input_img_LR_masked)
        else:
            raise ValueError('error')

        img_synth = img_fc * (1 - self._input_mask) + self._input_mask * self._input_img_LR
        _, _, _, img_fine_sr = self._SR.forward(img_synth)

        d_fake_img_prob = self._global_d.forward(img_fine_sr.detach())
        self._loss_d_fake = misc.compute_loss_d(d_fake_img_prob, False) * self._opt.lambda_global_d_prob

        d_real_img_prob = self._global_d.forward(self._input_img_SR)
        self._loss_d_real = misc.compute_loss_d(d_real_img_prob, True) * self._opt.lambda_global_d_prob

        return self._loss_d_real + self._loss_d_fake, img_fine_sr

    def _gradinet_penalty_d(self, synth_img, gt_img, discrinimator):

        alpha = torch.rand(self._bs, 1, 1, 1).expand_as(gt_img).to(self._device)
        interpolated = alpha * gt_img.data + (1 - alpha) * synth_img.data
        interpolated.requires_grad = True
        interpolated_prob = discrinimator.forward(interpolated)

        grad = torch.autograd.grad(outputs=interpolated_prob,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpolated_prob.size()).to(self._device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        self._loss_d_gp = torch.mean((grad_l2norm - 1) ** 2)

        return self._loss_d_gp

    def _inpainting_loss(self, img_synth, img_fc, img_gt):

        target = (1 - self._input_mask) * img_gt
        target = target.detach()
        loss_hole = misc.compute_loss_l1((1 - self._input_mask) * img_fc, target)

        target = self._input_mask * img_gt
        target = target.detach()
        loss_vaild = misc.compute_loss_l1(self._input_mask * img_fc, target)

        rec_img_feature = self._vgg(img_fc)
        synth_img_feature = self._vgg(img_synth)
        gt_img_feature = self._vgg(img_gt)

        loss_sty = 0
        loss_per = 0

        for i in range(len(rec_img_feature)):

            loss_sty += misc.compute_loss_l1(
                misc.compute_loss_gram_matrix(rec_img_feature[i]),
                misc.compute_loss_gram_matrix(gt_img_feature[i]))

            loss_sty += misc.compute_loss_l1(
                misc.compute_loss_gram_matrix(synth_img_feature[i]),
                misc.compute_loss_gram_matrix(gt_img_feature[i]))

            loss_per += misc.compute_loss_l1(rec_img_feature[i], gt_img_feature[i])
            loss_per += misc.compute_loss_l1(synth_img_feature[i], gt_img_feature[i])

        loss_synth_smooth = misc.compute_loss_smooth(img_synth)

        return loss_hole, loss_vaild, loss_sty, loss_per, loss_synth_smooth

    def get_current_errors(self):

        loss_dict = OrderedDict([('loss_1_hole', self._loss_1_hole.item()),
                                 ('loss_1_vaild', self._loss_1_vaild.item()),
                                 ('loss_1_sty', self._loss_1_sty.item()),
                                 ('loss_1_per', self._loss_1_per.item()),
                                 ('loss_1_synth_smooth', self._loss_1_synth_smooth.item()),

                                 ('loss_2_landmark', self._loss_2_landmark.item()),
                                 ('loss_2_face_parsing', self._loss_2_face_parsing.item()),
                                 ('loss_2_coarse', self._loss_2_coarse.item()),

                                 ('loss_2_per', self._loss_2_per.item()),
                                 ('loss_2_fine', self._loss_2_fine.item()),

                                 ('loss_g_global_adv', self._loss_g_global_adv.item()),

                                 ('loss_d_global_adv', self._loss_d_global_adv.item()),
                                 ('loss_d_global_adv_gp', self._loss_d_global_adv_gp.item()),
                                 ])

        return loss_dict

    def get_current_scalars(self):

        return OrderedDict([('lr_g', self._current_lr_g), ('lr_d', self._current_lr_d)])

    def get_current_visuals(self):

        visuals = OrderedDict()

        visuals['batch_img_mask'] = self._vis_batch_mask
        visuals['batch_img_LR_masked'] = self._vis_batch_img_LR_masked
        visuals['batch_img_LR_fc'] = self._vis_batch_img_LR_fc
        visuals['batch_img_LR_syn'] = self._vis_batch_img_LR_syn
        visuals['batch_img_LR'] = self._vis_batch_img_LR

        visuals['batch_img_coarse'] = self._vis_batch_img_coarse

        visuals['batch_img_landmark'] = self._vis_batch_img_landmark
        visuals['batch_img_landmark_GT'] = self._vis_batch_img_landmark_GT

        visuals['batch_img_face_parsing'] = self._vis_batch_img_face_parsing
        visuals['batch_img_face_parsing_GT'] = self._vis_batch_img_face_parsing_GT

        visuals['batch_img_fine'] = self._vis_batch_img_fine
        visuals['batch_img_SR'] = self._vis_batch_img_SR

        return visuals

    def save(self, label):

        if self._opt.fix_fc:
            self._save_network(self._SR, 'SR', label)
            self._save_optimizer(self._opti_g, 'opti_SR', label)
        else:

            self._save_network(self._FC, 'FC', label)
            self._save_network(self._SR, 'SR', label)
            self._save_optimizer(self._opti_g, 'opti_FCSR', label)

        self._save_network(self._global_d, 'global_d', label)
        self._save_optimizer(self._opti_global_d, 'opti_global_d', label)

    def load(self):

        load_epoch = self._opt.load_epoch

        if self._opt.fix_fc:
            self._load_network(self._SR, 'SR', load_epoch)
        else:
            self._load_network(self._FC, 'FC', load_epoch)
            self._load_network(self._SR, 'SR', load_epoch)

        if self._is_train:

            self._load_network(self._global_d, 'global_d', load_epoch)
            self._load_optimizer(self._opti_global_d, 'opti_global_d', load_epoch)

            if self._opt.fix_fc:
                self._load_optimizer(self._opti_g, 'opti_SR', load_epoch)

            else:
                self._load_optimizer(self._opti_g, 'opti_FCSR', load_epoch)

    def update_learning_rate(self):

        lr_decay = self._opt.lr_g / self._opt.nepochs_decay
        self._current_lr_g -= lr_decay

        lr_decay_g = self._opt.lr_g / self._opt.nepochs_decay
        self._current_lr_g -= lr_decay_g
        for param_group in self._opti_g.param_groups:
            param_group['lr'] = self._current_lr_g
        print('update G learning rate: %f -> %f' % (self._current_lr_g + lr_decay_g, self._current_lr_g))

        lr_decay_d = self._opt.lr_d / self._opt.nepochs_decay
        self._current_lr_d -= lr_decay_d
        for param_group in self._opti_global_d.param_groups:
            param_group['lr'] = self._current_lr_d
        print('update global D learning rate: %f -> %f' % (self._current_lr_d + lr_decay_d, self._current_lr_d))
