import argparse
from sampling import get_supertransformer_config
from utils import (
    millify,
    calculate_params_from_config,
    read_json,
)
import os

os.environ["PATH"] = "../"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--subtransformer_config_path",
        type=str,
        default=None,
        help=f"The path to a subtransformer configration",
    )
    parser.add_argument(
        "--mixing",
        type=str,
        required=True,
        help=f"specifies how to mix the tokens in bertlayers",
        choices=["attention", "gmlp", "fnet", "mobilebert", "bert-bottleneck"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-cased",
        help="Name of the huggingface model",
    )
    args = parser.parse_args()

    global_config = get_supertransformer_config(args.model_name, mixing=args.mixing)
    # hardfixed for now so that we can get the params without any task specific config
    global_config.num_labels = 2

    if args.subtransformer_config_path is not None:
        subtransformer_config = read_json(args.subtransformer_config_path)
        for key, value in subtransformer_config.items():
            # update global_config with attributes of subtransformer_config
            setattr(global_config, key, value)
        params = calculate_params_from_config(
            global_config,
            scaling_laws=False,
            add_output_emb_layer=False,
            merged_bottleneck=True,
        )
        print("==================================================================")
        print(
            f"Number of parameters in custom config is {params} -->  {millify(params)}"
        )
        print("==================================================================")
