import os
import numpy as np

from RandWireNN_train import prepare, train_faster_rcnn, store_eval_model_with_native_udf
# from RandWireNN_eval import compute_test_set_aps, FasterRCNN_Evaluator
from utils.config_helpers import merge_configs

def get_configuration():
    # load configs for base network, random graph model and data set
    from RandWireNN_config import cfg as network_cfg

    from utils.configs.WS_config import cfg as graph_cfg
    # from utils.configs.BA_config import cfg as graph_cfg
    # from utils.configs.WS_config import cfg as graph_cfg

    from utils.configs.CIFAR100 import cfg as dataset_cfg
    # for the CIFAR10 data set use:     from utils.configs.ImageNet_config import cfg as dataset_cfg
    # for the ImageNet data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg

    return merge_configs([network_cfg, graph_cfg, dataset_cfg])

# trains and evaluates a Fast R-CNN model.
if __name__ == '__main__':
    cfg = get_configuration()

    prepare(cfg)
    
    model = CNN(cfg)
    
    # load
    model.load_state_dict(torch.load("./output/model/049_000000.cpt"))
    model.cuda()
    
    # train and test
    trained_model = train_faster_rcnn(cfg)

    eval_results = compute_test_set_aps(trained_model, cfg)
        
    # write AP results to output
    for class_name in eval_results: print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))

    # Plot results on test set images
    if cfg.VISUALIZE_RESULTS:
        num_eval = cfg["DATA"].NUM_TEST_IMAGES
        results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
        print(results_folder)

        evaluator = FasterRCNN_Evaluator(trained_model, cfg)
        plot_test_set_results(evaluator, num_eval, results_folder, cfg)

    if cfg.STORE_EVAL_MODEL_WITH_NATIVE_UDF:
        store_eval_model_with_native_udf(trained_model, cfg)

