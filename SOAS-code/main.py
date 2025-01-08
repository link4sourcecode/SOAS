from util.config_for_all import parse_args
from util.my_util import get_model
from util.st_generation import steering_vector_generation
from util.st_deployment import steering_vector_deployment, defense_layer_selection
from util.evaluation import defense_evaluation

def main(args):
    # loading model
    model, tokenizer, num_of_layers, emb_dim, projection_matrix = get_model()

    # defense layer selection
    defense_layer_idx, _ = defense_layer_selection(args, model, num_of_layers, projection_matrix)

    # loading steering vector
    steering_vec = steering_vector_generation(args, model, tokenizer, num_of_layers, emb_dim)

    # steering vector generation
    steering_vector_deployment(args, defense_layer_idx)

    # defense evaluation
    defense_evaluation(args, model, tokenizer)

if __name__ == "__main__":
    user_args = parse_args()
    main(user_args)