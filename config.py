import argparse


def setting_params(mode: str,
                   description: str,
                   data_path: str,
                   label_info_path: str,
                   use_cuda: bool,
                   multi_gpu: bool,
                   num_epochs: int = 500,
                   train_batch_size: int = 4,

                   eval_batch_size: int = 4,
                   save_model: bool = False,
                   load_saved_model: bool = False,
                   path_saved_model: str = '',
                   use_frame: tuple = (6, 24, 1),
                   save_image: bool = False,
                   save_roc_curve: bool = False):

    if save_model and not mode == 'train':
        raise Exception("'save_model' is available only in 'train' mode")
    if mode == 'test' and not load_saved_model:
        raise Exception("When in 'test' mode, a saved model is required. <\n>"
                        "Set the 'load_saved_mode'=True")
    if load_saved_model and not path_saved_model:
        raise Exception("If you want to load a saved model, you need to write 'path_saved_model'")

    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, default=description,
                        help='A brief description of the code you are currently running')
    parser.add_argument('--mode', type=str, default=mode, choices=['train', 'test'], help='mode: train / test')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='dataset path : '
                             'dataset dir'
                             '├── train'
                             '└── eval')
    parser.add_argument('--label_info_path', type=str, default=label_info_path, help='label_info.csv')

    # fixed seed & gpu setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_cuda', type=bool, default=use_cuda)
    parser.add_argument('--multi_gpu', type=bool, default=multi_gpu,
                        help='It is recommended to set it to True when the number of available gpus is 2 or more.')

    # flag - save model & load saved model & save_image
    parser.add_argument('--th_epoch', type=int, default=0)
    parser.add_argument('--save_model', type=bool, default=save_model)
    parser.add_argument('--load_saved_model', type=bool, default=load_saved_model)
    parser.add_argument('--path_saved_model', type=str, default=path_saved_model)
    parser.add_argument('--save_image', type=bool, default=save_image)
    parser.add_argument('--save_roc_curve', type=bool, default=save_roc_curve)

    # data preprocess params
    parser.add_argument('--use_frame', type=tuple, default=use_frame,
                        help='(start, end, step)'
                             '(6, 24, 1)=[60s, 70s, ... , 220s, 230s]'
                        )
    parser.add_argument('--img_use_hsv', type=str, default='parallel', choices=['not_use', 'hsv', 'parallel'],
                        help='>> mode_list <<'
                             'not_use : only rgb'
                             'hsv : only hsv'
                             'parallel : both rgb & hsv'
                        )
    parser.add_argument('--use_noise', type=bool, default=True, help='flag - use gaussian noise')

    # training params
    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--train_batch_size', type=int, default=train_batch_size)
    parser.add_argument('--eval_batch_size', type=int, default=eval_batch_size)
    # parser.add_argument('--step_batch', type=int, default=1)

    # model params
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--criterion', type=str, default='l1', choices=['l1', 'mse'])
    parser.add_argument('--latent_size', type=int, default=1024)

    parser.add_argument('--img_embedding_size', type=int, default=1000, help='if use pretrained, only 1000')
    parser.add_argument('--img_encoder', type=str, default='d-121', choices=['r-18', 'r-34','r-50','d-121'])
    parser.add_argument('--seq_model', type=str, default='lstm', choices=['lstm', 'gru'])

    parser.add_argument('--pretrained_reg', type=bool, default=False, help='load pretrained Regression model')
    parser.add_argument('--reg_model_path', type=str, default='', help='pretrained Regression model path')
    parser.add_argument('--pretrained', type=bool, default=True, help='load pretrained CNN model')

    return parser.parse_args()
