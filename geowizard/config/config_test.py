# 定义超参数
hyperparameters = {
    'is_run_script': True,
    'start': 0,
    'num_images_to_load': 300,  # Example number, change as needed
    'is_visualize': True,
    'is_save_metric': True,
    'prefix': "_pred_colored",  # 深度图存的前缀/后缀，例如depth_aov_
    'pred_folder': 'Saved_Depth',
    # -----------重要基本参数-----------
    'model_name': 'GeoWizard-EndoSlam-Unity',
    'vis_path': 'Saved_visulaization/',
    'pred_depth_img_path': 'Saved_depth_data/',
    'metrics_path': 'Saved_metrics_data/',
    'checkpoint_path': 'lightning_logs/version_4/checkpoints/epoch=932-step=506619.ckpt',
    'CUDA': "1",
    # -----------数据集相关参数-----------
    'gt_folder': r"/mnt/share/toky/Datasets/Endoslam_Unity_Colon/Pixelwise_Depths/",
    'input_folder': '/mnt/share/toky/Datasets/EndoDepth-Diffusion/EndoSlam-Unity/eval/',
    # [''/mnt/share/toky/Datasets/EndoDepth-Diffusion/EndoSlam-Phantom/eval/'',
    # '/mnt/share/toky/Datasets/EndoDepth-Diffusion/EndoMapper-Clinical-Seq/eval/',
    # '/mnt/share/toky/Datasets/ControlNet_Dataset/test2171/dpt_predictit_depth/',
    # '/mnt/share/toky/Datasets/SERV-Depth/rgb/',
    # '/mnt/share/toky/Datasets/ControlNet_Dataset/test2171/dpt_canny_merge/',
    # '/mnt/share/toky/Datasets/ControlNet_Dataset/test2171/dpt_predictit_depth/',
    # '/mnt/share/toky/Datasets/ControlNet_Dataset/scared/test2/source/',
    # '/mnt/share/toky/Datasets/ControlNet_Dataset/test',
    # '/mnt/share/toky/Datasets/ControlNet_Dataset/scared/test/source/']
    'saved_depth': 'saved_depth',

}
