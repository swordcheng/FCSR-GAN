from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)

        self._parser.add_argument('--print_freq_s', type=int, default=30,
                                  help='frequency of showing training results on console')
        self._parser.add_argument('--display_freq_s', type=int, default=180,
                                  help='frequency [s] of showing training results on screen')
        self._parser.add_argument('--save_latest_freq_s', type=int, default=3600,
                                  help='frequency of saving the latest results')

        self._parser.add_argument('--nepochs_no_decay', type=int, default=50,
                                  help='# of epochs at starting learning rate')
        self._parser.add_argument('--nepochs_decay', type=int, default=10,
                                  help='# of epochs to linearly decay learning rate to zero')

        self._parser.add_argument('--n_threads_train', type=int, default=0, help='')
        self._parser.add_argument('--n_threads_test', type=int, default=0, help='')

        self._parser.add_argument('--batch_size', type=int, default=8, help='')
        self._parser.add_argument('--freeze_enc_bn', type=bool, default=False, help='')

        self._parser.add_argument('--upsample', type=bool, default=True, help='')
        self._parser.add_argument('--fix_fc', type=bool, default=False, help='')
        self._parser.add_argument('--fc_pretrain', type=bool, default=False, help='')

        self._parser.add_argument('--fc_pretrain_model_path', type=str,
                                  default='/home/jccai/code/pytorch/FCSRNet-saved/checkpoints/pretrain_1/pconv_16X16_X8.pth',
                                  help='which epoch to load? set to -1 to use latest cached model')

        self._parser.add_argument('--lr_g', type=float, default=0.0001, help='initial learning rate for adam')
        self._parser.add_argument('--g_adam_b1', type=float, default=0.5, help='beta1 for adam')
        self._parser.add_argument('--g_adam_b2', type=float, default=0.999, help='beta2 for adam')

        self._parser.add_argument('--lr_d', type=float, default=0.0001, help='initial learning rate for adam')
        self._parser.add_argument('--d_adam_b1', type=float, default=0.5, help='beta1 for adam')
        self._parser.add_argument('--d_adam_b2', type=float, default=0.999, help='beta2 for adam')

        self._parser.add_argument('--scale_factor', type=int, default=4, help='')
        self._parser.add_argument('--img_size', type=int, default=128, help='')
        self._parser.add_argument('--heatmap_size', type=int, default=64, help='')

        self._parser.add_argument('--train_G_every_n_iterations', type=int, default=5, help='')

        self._parser.add_argument('--lambda_loss_1_hole', type=float, default=6, help='')
        self._parser.add_argument('--lambda_loss_1_vaild', type=float, default=1, help='')
        self._parser.add_argument('--lambda_loss_1_sty', type=float, default=120, help='')
        self._parser.add_argument('--lambda_loss_1_per', type=float, default=0.05, help='')
        self._parser.add_argument('--lambda_loss_1_synth_smooth', type=float, default=0.1, help='')

        self._parser.add_argument('--lambda_loss_2_coarse', type=float, default=1, help='')
        self._parser.add_argument('--lambda_loss_2_landmark', type=float, default=1, help='')
        self._parser.add_argument('--lambda_loss_2_parsing', type=float, default=0.1, help='')
        self._parser.add_argument('--lambda_loss_2_per', type=float, default=0.05, help='')
        self._parser.add_argument('--lambda_loss_2_fine', type=float, default=1, help='')

        self._parser.add_argument('--lambda_global_d_prob', type=float, default=0.01, help='')
        self._parser.add_argument('--lambda_global_d_gp', type=float, default=0.1, help='')

        self._parser.add_argument('--load_epoch', type=int, default=-1,
                                  help='which epoch to load? set to -1 to use latest cached model')

        self._parser.add_argument('--show_max', type=int, default=8, help='')

        self._parser.add_argument('--name', type=str, default='experiment_face_pconv_fsrnet_X4',
                                  help='name of the experiment.')
        self._parser.add_argument('--model', type=str, default='pconv+fsrnet', help='')
        self._parser.add_argument('--sr_module', type=str, default='fsrnet', help='')
        self._parser.add_argument('--fc_module', type=str, default='pconv', help='')

        self.is_train = True
