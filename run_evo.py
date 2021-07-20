import accelerate
from accelerate import accelerator
from hf_evo import Evosearch
from hf_lat_lgbm import LatencyPredictor
from accelerate import Accelerator
import os


def get_num_params(config):
    N = config['encoder']['encoder_layer_num']
    d_e = config['encoder']['encoder_embed_dim']
    d_ff_list = config['encoder']['encoder_ffn_embed_dim']
    num_params = 4*d_e*d_e*N
    for d_ff in d_ff_list:
        num_params += 2*d_e*d_ff
    return num_params


if __name__ == "__main__":
    accel = Accelerator(cpu=False, fp16=True)

    # Calculate Max Latency:
    predictor = LatencyPredictor(ckpt_path='./latency_dataset/ckpts/lgb_cpu_985.txt')
    # predictor = LatencyPredictor(ckpt_path='./latency_dataset/ckpts/lgb_724.txt')
    predictor.load_ckpt()
    max_config = {
        'encoder': {
            'encoder_embed_dim': 768,
            'encoder_layer_num': 12,
            'encoder_ffn_embed_dim': [3072]*12,
            'encoder_self_attention_heads': [12]*12,
        }
    }
    predict = predictor.predict_lat(max_config)
    max_latency = predict

    # Calculate Min Latency:
    predictor = LatencyPredictor(ckpt_path='./latency_dataset/ckpts/lgb_cpu_985.txt')
    # predictor = LatencyPredictor(ckpt_path='./latency_dataset/ckpts/lgb_724.txt')
    predictor.load_ckpt()
    min_config = {
        'encoder': {
            'encoder_embed_dim': 360,
            'encoder_layer_num': 2,
            'encoder_ffn_embed_dim': [512]*2,
            'encoder_self_attention_heads': [6]*2,
        }
    }
    predict = predictor.predict_lat(min_config)
    min_latency = predict

    # Set the latencies cap list:
    num_latencies = 5
    common_difference = (max_latency-min_latency)/num_latencies
    latency_thresholds = []
    for i in range(1, num_latencies+1):
        latency_thresholds.append(min_latency+common_difference*i)

    accel.print(f'Min latency = {min_latency}\nMax latency = {max_latency}\nlatency thresholds = {latency_thresholds}')

    # Setup tasks and evosearch params:
    tasks = ['sst2', 'mrpc', 'qqp', 'qnli', 'rte']
    pop_sizes = {'qnli':24, 'sst2':30, 'rte': 30, 'mrpc': 30, 'qqp':24}
    par_sizes = {'qnli':8, 'sst2':10, 'rte': 10, 'mrpc': 10, 'qqp': 8}
    mut_sizes = {'qnli': 8, 'sst2': 10, 'rte': 10, 'mrpc': 10, 'qqp': 8}
    cro_sizes = {'qnli': 8, 'sst2': 10, 'rte': 10, 'mrpc': 10, 'qqp': 8}
    num_runs  = {'qnli': 5, 'sst2': 5, 'rte': 5, 'mrpc': 5, 'qqp': 5}


    os.environ["TOKENIZERS_PARALLELISM"]='false'
    search_space = {
        "encoder_embed_dim": [360, 480, 540, 600, 768],
        "encoder_layer_num": [2, 4, 6, 8, 10, 12],
        "encoder_ffn_embed_dim": [512, 1024, 2048, 3072],
        "encoder_self_attention_heads": [6, 8, 10, 12],
    }
    save_directory = 'evo_search_results'

    for task in tasks:
        accel.print(f'Task: {task}')
        for idx, latency_cap in enumerate(latency_thresholds):
            accel.print(f'latency cap {idx} running...')
            runner = Evosearch(
                12,
                pop_sizes[task],
                par_sizes[task],
                mut_sizes[task],
                cro_sizes[task],
                search_space,
                latency_cap,
                task,
                0.3,
                num_runs[task],
                'checkpoints/'+task+'/pytorch_model.bin',
                accel = accel
            )
            config, max_acc, config_latency = runner.run_evo_search()
            num_params = get_num_params(config)
            accel.print(f'Number of parameters of best config: {num_params}')
            if accel.is_main_process:
                filename = f'{save_directory}/{task}/evo_dump_lat_{idx}.txt'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                open('file.txt', 'w').close()
                file = open(filename, 'w')
                file.write(str(config)+'\n'+str(max_acc)+'\n'+str(config_latency)+'\n'+str(num_params))
                

        

