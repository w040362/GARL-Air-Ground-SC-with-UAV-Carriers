from util import *
from gpu_manager import GPUManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='KAIST', help='the name of dataset (KAIST or UCLA)')
    parser.add_argument('--method', type=str, default='GARL', help='the name of method (GARL)')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    args = parser.parse_args()

    dataset_conf = importlib.import_module('datasets.' + args.dataset + '.conf_temp').DATASET_CONF
    env_conf = importlib.import_module('env.conf_temp').ENV_CONF
    method_conf = importlib.import_module('methods.' + args.method + '.conf_temp_' + args.dataset).METHOD_CONF
    log_conf = importlib.import_module('log.conf_temp').LOG_CONF

    log_conf['dataset_name'] = args.dataset
    log_conf['method_name'] = args.method

    global_dict_init()
    set_global_dict_value('dataset_conf', dataset_conf)
    set_global_dict_value('env_conf', env_conf)
    set_global_dict_value('method_conf', method_conf)
    set_global_dict_value('log_conf', log_conf)

    fix_random_seed(get_global_dict_value('method_conf')['seed'])

    # choose best gpu
    gm = GPUManager()
    method_conf['gpu_id'] = gm.auto_choice(mode=0)

    if args.mode == 'train':
        main = importlib.import_module('methods.' + args.method + '.main')
        main.main()
    elif args.mode == 'test':
        main_test = importlib.import_module('methods.' + args.method + '.main_test')
        main_test.main()
