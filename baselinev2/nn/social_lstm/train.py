import argparse

from baselinev2.nn.social_lstm.model import LINEAR, LSTM


def bool_flag(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ("yes", "true", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value excepted!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Trajectory Prediction Basics")

    # Configs for Model
    parser.add_argument("--model_name", default="", type=str, help="Define model name for saving")
    parser.add_argument("--model_type", default="lstm", type=str,
                        help="Define type of model. Choose either: linear, lstm or social-lstm")
    parser.add_argument("--save_model", default=True, type=bool_flag, help="Save trained model")
    parser.add_argument("--nl_ADE", default=False, type=bool_flag, help="Use nl_ADE")
    parser.add_argument("--load_model", default=False, type=bool_flag, help="Specify whether to load existing model")
    parser.add_argument("--lstm_pool", default=False, type=bool_flag, help="Specify whether to enable social pooling")
    parser.add_argument("--pooling_type", default="social_pooling", type=str, help="Specify pooling method")
    parser.add_argument("--neighborhood_size", default=10.0, type=float, help="Specify neighborhood size to one side")
    parser.add_argument("--grid_size", default=10, type=int, help="Specify grid size")
    parser.add_argument("--args_set", default="", type=str,
                        help="Specify predefined set of configurations for respective model. "
                             "Choose either: lstm or social-lstm")

    # Configs for data-preparation
    parser.add_argument("--dataset_name", default="to_be_defined", type=str, help="Specify dataset")
    parser.add_argument("--dataset_type", default="square", type=str,
                        help="Specify dataset-type. For real datasets choose: 'real'. "
                             "For synthetic datasets choose either 'square' or 'rectangle'")
    parser.add_argument("--obs_len", default=8, type=int, help="Specify length of observed trajectory")
    parser.add_argument("--pred_len", default=12, type=int, help="Specify length of predicted trajectory")
    parser.add_argument("--data_augmentation", default=True, type=bool_flag,
                        help="Specify whether or not you want to use data augmentation")
    parser.add_argument("--batch_norm", default=False, type=bool_flag, help="Batch Normalization")
    parser.add_argument("--max_num", default=1000000, type=int, help="Specify maximum number of ids")
    parser.add_argument("--skip", default=20, type=int, help="Specify skipping rate")
    parser.add_argument("--PhysAtt", default="", type=str, help="Specify physicalAtt")
    parser.add_argument("--padding", default=True, type=bool_flag, help="Specify if padding should be active")
    parser.add_argument("--final_position", default=False, type=bool_flag,
                        help="Specify whether final positions of pedestrians should be passed to model or not")

    # Configs for training, validation, testing
    parser.add_argument("--batch_size", default=32, type=int, help="Specify batch size")
    parser.add_argument("--wd", default=0.03, type=float, help="Specify weight decay")
    parser.add_argument("--lr", default=0.001, type=float, help="Specify learning rate")
    parser.add_argument("--encoder_h_dim", default=64, type=int, help="Specify hidden state dimension h of encoder")
    parser.add_argument("--decoder_h_dim", default=32, type=int, help="Specify hidden state dimension h of decoder")
    parser.add_argument("--emb_dim", default=32, type=int, help="Specify dimension of embedding")
    parser.add_argument("--num_epochs", default=250, type=int, help="Specify number of epochs")
    parser.add_argument("--dropout", default=0.0, type=float, help="Specify dropout rate")
    parser.add_argument("--num_layers", default=1, type=int, help="Specify number of layers of LSTM/Social LSTM Model")
    parser.add_argument("--optim", default="Adam", type=str,
                        help="Specify optimizer. Choose either: adam, rmsprop or sgd")

    # Get arguments
    args = parser.parse_args()

    print("Linear")
    print(LINEAR(args))

    print("LSTM")
    print(LSTM(args))
