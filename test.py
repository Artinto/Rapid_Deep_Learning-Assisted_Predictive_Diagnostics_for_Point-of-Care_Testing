from config import setting_params
from main import main


if __name__ == '__main__':
    args = setting_params(
        mode='test',
        description='latent:1024+pretrain-r50+lstm',
        data_path='./dataset/Standard_sample',
        label_info_path='./dataset/label_info.csv',
        use_cuda=True,
        multi_gpu=False,
        eval_batch_size=1,
        load_saved_model=True,
        path_saved_model='./log/train/log_time_series_230807_12-09/model_save/best_density_model.pt',
        save_image=True,
        save_roc_curve=False,
        use_frame=(0, 12, 1)
    )

    main(args)
