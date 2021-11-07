import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_data_path',
        type=str,
        help='input data file path',
        default='data/processed-data/inmasker_tokenised_data_mm_toy.jsonl',
    )
    parser.add_argument(
        '--label_vocab_file_path',
        type=str,
        help='file path for the vocab',
        default='data/processed-data/label_vocab_mm_toy.json',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Outpur dir for the model',
        default='/tmp/',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='batch size',
        default=32,
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        help='Num of workers for the dataloader',
        default=4,
    )
    parser.add_argument(
        '--model_type',
        choices=['inmasker', 'simple_classifier'],
        default='simple_classifier',
    )
    parser.add_argument(
        '--dataset',
        choices=['mmst21pv_sample', 'bc5cdr'],
        default='mmst21pv_sample',
        help='Dataset to choose for model training'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        help='Path for the model checkpoint for testing',
        default=None,
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Whether to test the model',
        default=False
    )
    parser.add_argument(
        '--classifier_dropout',
        type=float,
        help='Val of classifier dropout',
        default=0.2,
    )
    parser.add_argument(
        '--pretrained_bert_path',
        type=str,
        help='Path to pretrained bert, default not using pretrained bert',
        default=None,
    )
    parser.add_argument(
        '--encoder_config_file_path',
        type=str,
        help='Path to transformer encoder config, default saved in repo.',
        default='runners/utils/small_transformer_config.json',
    )
    parser.add_argument(
        '--lambda',
        type=float,
        help='Lambda value',
        default=0.1,
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        '--mask_token_id',   # TODO remove this as an argument and find automatically in train script
        type=int,
        help='Mask token id, only works for transformer or SciBERT',
        default=105,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate'
    )
    parser.add_argument(
        '--grad_clip_val',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=20,
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=6,
    )
    parser.add_argument(
        '--monitor_quantity',
        type=str,
        help='Monitor quantity to use for early stopping and checkpointing',
        default='val_loss',
    )
    parser.add_argument(
        '--token_importance_method',
        choices=['basic', 'attention', 'basic_with_frozen_layer'],
        help='Token importance method to use',
        default='basic',
    )
    parser.add_argument(
        '--seed_val',
        type=int,
        help='Random seed val',
        default=12,
    )
    parser.add_argument(
        '--warmup_proportion',
        type=float,
        help='warmup proportion to use for linear scheduler',
        default=0.2,
    )
    parser.add_argument(
        '--full_multi_task_learning',
        action='store_true',
        help='Whether actor and two critics should share the weights',
        default=False
    )
    parser.add_argument(
        '--critic_multi_task_learning',
        action='store_true',
        help='Whether two critics should share the weights',
        default=False
    )
    parser.add_argument(
        '--simple_classifier_l2_penalty',
        action='store_true',
        help='Whether two critics should share the weights',
        default=False
    )
    return parser.parse_args()
