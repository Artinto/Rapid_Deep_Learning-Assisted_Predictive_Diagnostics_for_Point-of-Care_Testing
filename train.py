from config import setting_params
from main import main


if __name__ == '__main__':
    args = setting_params(
        mode='train',
        description='latent:1024+pretrain-r50+lstm',
        data_path='./dataset/Standard_sample',
        label_info_path='./dataset/label_info.csv',
        use_cuda=True,
        multi_gpu=True,
        num_epochs=500,
        train_batch_size=16,
        eval_batch_size=2,
        save_model=True,
        use_frame=(0, 12, 1)
    )

    main(args)
