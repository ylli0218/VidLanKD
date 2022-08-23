import argparse


def process_args():
    parser = argparse.ArgumentParser()

    # Datasets
    parser.add_argument(
        "--train_data_file", default=None, type=str,
        help="The input training data file (a text file).")
    parser.add_argument(
        "--eval_data_file", default=None, type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    # Data loader
    parser.add_argument("--col_data", action="store_true", help="Using the specific dataset object in data.py")
    parser.add_argument("--split_sent", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the training dataset")
    parser.add_argument(
        "--block_size", default=-1, type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )

    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument(
        "--output_dir", type=str,
        help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument(
        "--overwrite_output_dir", action="store_true",
        help="Overwrite the content of the output directory")

    # Model types
    parser.add_argument(
        "--model_type", type=str, help="The model architecture to be trained or fine-tuned.",)
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir")
    parser.add_argument(
        "--model_name_or_path", default=None, type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",)
    parser.add_argument(
        "--config_name", default=None, type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",)
    parser.add_argument(
        "--tokenizer_name", default=None, type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",)
    parser.add_argument(
        "--cache_dir", default=None, type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",)
    parser.add_argument(
        "--overwrite_cache", action="store_true",
        help="Overwrite the cached training and evaluation sets")

    # MLM tasks
    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument(
        "--voken_ratio", type=float, default=1., help="The ratio of mlm loss in the total loss.")

    # VLM related params
    parser.add_argument("--teacher_dir", type=str)
    parser.add_argument("--do_kd1_objective", action='store_true', help='Will do mmd loss')
    parser.add_argument("--do_kd2_objective", action='store_true', help='Will do crd loss')
    parser.add_argument("--margin", default=0.5, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--no_nst_transpose", action='store_true', help='there is a transpose operation before nst function, test whether it is necessary')
    
    # Batch Size and Training Steps
    parser.add_argument("--seed", type=int, default=95, help="random seed for initialization")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--save_steps", type=int, default=10000,
                        help="Save every X steps.")

    # Optimizer
    parser.add_argument("--lamb", action="store_true", help='Use the LAMB optimizer in apex')
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0., type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # Distributed Training
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--nr", type=int, default=0)

    # Half Precision
    parser.add_argument(
        "--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument(
        "--fp16_opt_level", type=str, default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",)

    args = parser.parse_args()
    return args
