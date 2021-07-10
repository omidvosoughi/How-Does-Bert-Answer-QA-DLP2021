from typing import List, Tuple

from sklearn.decomposition import PCA

from .plotting import Token2DVector, TokenPlotter, TokenLabel
from .pretrained_model_loader import QAModel
from .sample_wrapper import QASample


class Visualizer:
    def visualize_hidden_states(self, model: QAModel, sample_file_path: str):
        """
        Given a model and a sample visualize the output of the hidden states in a 2D plot
        by reducing the feature dimension with PCA.
        """
        sample = QASample.from_json_file(sample_file_path)
        prediction, hidden_states, features = model.predict(sample)
        tokens: List = features.tokens
        print(tokens)

        question_indices: Tuple = self.__get_question_indices(tokens)
        prediction_indices: Tuple = (prediction["start_index"], prediction["end_index"])

        token_labels: List[TokenLabel] = self.__get_labels_for_tokens(
            tokens, question_indices, prediction_indices, features.sup_ids
        )
        for layer_index, layer in enumerate(hidden_states):
            token_vectors: List = layer[0][:len(tokens)]
            layer_reduced: List = (
                PCA(n_components=2).fit_transform(token_vectors).transpose()
            )
            token_vectors = []
            for token_index, value in enumerate(
                layer_reduced[0]
            ):  # iterate over x-values

                token_vector = Token2DVector(
                    x=value,
                    y=layer_reduced[1][token_index],
                    token=tokens[token_index],
                    label=token_labels[token_index],
                )
                token_vectors.append(token_vector)

            plot_title = "Layer {}".format(layer_index)

            token_plotter = TokenPlotter(
                vectors=token_vectors, title=plot_title, output_path=None
            )
            token_plotter.plot()

    def __get_labels_for_tokens(
        self, tokens, question_pos, prediction_pos, sup_facts_pos
    ):

        token_labels = []

        for token_pos, _ in enumerate(tokens):

            if prediction_pos[0] <= token_pos <= prediction_pos[1]:
                token_labels.append(TokenLabel.PREDICTION)
                continue
            if question_pos[0] <= token_pos <= question_pos[1]:
                token_labels.append(TokenLabel.QUESTION)
                continue

            is_supporting_fact_token = False
            for sup_fact_pos in sup_facts_pos:
                if sup_fact_pos[0] <= token_pos <= sup_fact_pos[1]:
                    is_supporting_fact_token = True
                    break

            if is_supporting_fact_token:
                token_labels.append(TokenLabel.SUP_FACT)
            else:
                token_labels.append(TokenLabel.DEFAULT)

        return token_labels

    def __get_question_indices(self, tokens: List) -> Tuple:
        """
        Get start and end position for tokens before the [SEP] token, which are the question tokens in QA.
        """
        sep_token = "[SEP]"
        start_index = 1  # skip first token as it is [CLS]
        end_index = tokens.index(sep_token) - 1
        return start_index, end_index
