import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Defense against Extraction Attack via Activation Steering")

    # ======================== Model Config Preparation =========================
    parser.add_argument("--model", type=str, default='gpt-j-6b', help="load HuggingFace model")
    parser.add_argument("--device", type=str, default='cuda:0', help="default gpu index")
    # ======================== Data Preparation =========================
    parser.add_argument("--prefix_root", type=str, default='./data/valid_prefix.npy', help="root for prefix")
    parser.add_argument("--suffix_root", type=str, default='./data/valid_suffix.npy', help="root for suffix")
    parser.add_argument("--chunk", type=int, default=1500, help="split the data")
    # ======================== SOAS hyperparameters =========================
    parser.add_argument("--trail", type=int, default=1, help="number of trails for generation")
    parser.add_argument("--bs", type=int, default=8, help="batch size")
    parser.add_argument("--generation_seed", type=int, default=100, help="check special prefix")
    parser.add_argument("--attack_trail", type=int, default=5, help="attempts for extraction attacks")
    parser.add_argument("--prefix_len", type=int, default=50, help="length for prefix")
    parser.add_argument("--suffix_len", type=int, default=50, help="length for suffix")
    parser.add_argument("--steering_strength", type=float, default=10.0, help="the strength of SOAS")

    return parser.parse_args()
