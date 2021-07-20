from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from typing import Tuple

class BertEdgeProbingSingleSpan(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.projection_size = 256
        self.hidden_size = 256


        self.bert = BertModel(config, add_pooling_layer=False)
        # Disable autograd for all parameters of the BertModel.
        for param in self.bert.parameters():
            param.requires_grad = False
        self.projection1 = nn.Linear(config.hidden_size, self.projection_size)

        self.attention1 = nn.Linear(self.projection_size, 1)

        self.linear1 = nn.Linear(self.projection_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.num_labels)

        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            span1s=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ):
        """Calculates the output of the net.

        The most important inputs are::
        input_ids:: tokenized sequence of shape (N, L, E)
            N=batch size, L=max sequence length, E=embedding dimension
        span1s:: masks for span1 of shape (N, L)
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        # outputs[0] has shape (N, L, E).
        span1 = outputs[0] * span1s.unsqueeze(-1)
        span1_projected = self.projection1(span1)
        # span1_projected has shape (N, L, 256), but is zero on all sequence elements not in the span.
        attention1 = self.attention1(span1_projected).squeeze()
        attention1 = F.softmax(attention1, 1).unsqueeze(2)
        mlp_input = torch.bmm(torch.transpose(span1_projected, 1, 2), attention1).squeeze(2)
        # mlp_input has shape (N, 256).
        mlp_output = self.sigmoid(self.linear1(mlp_input))
        mlp_output = self.sigmoid(self.linear2(mlp_output))
        mlp_output = self.sigmoid(self.output(mlp_output))

        return mlp_output


class BertEdgeProbingTwoSpan(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers
        self.projection_size = 256
        self.hidden_size = 256

        self.bert = BertModel(config, add_pooling_layer=False)
        # Disable autograd for all parameters of the BertModel.
        for param in self.bert.parameters():
            param.requires_grad = False
        self.projection1 = nn.Linear(config.hidden_size, self.projection_size)
        self.projection2 = nn.Linear(config.hidden_size, self.projection_size)

        self.attention1 = nn.Linear(self.projection_size, 1)
        self.attention2 = nn.Linear(self.projection_size, 1)

        self.linear1 = nn.Linear(2*self.projection_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.num_labels)


        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            span1s=None,
            span2s=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ):
        """Calculates the output of the net.

        The most important inputs are::
        input_ids:: tokenized sequence of shape (N, L, E)
            N=batch size, L=max sequence length, E=embedding dimension
        span1s:: masks for span1 of shape (N, L)
        span2s:: masks for span2 of shape (N, L)
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        # outputs[0] has shape (N, L, E)
        span1 = outputs[0] * span1s.unsqueeze(-1)
        span2 = outputs[0] * span2s.unsqueeze(-1)
        span1_projected = self.projection1(span1)
        span2_projected = self.projection2(span2)
        # span1_projected has shape (N, L, 256), but is zero on all sequence elements not in the span.
        attention1 = self.attention1(span1_projected).squeeze()
        attention2 = self.attention2(span2_projected).squeeze()
        attention1 = F.softmax(attention1, 1).unsqueeze(2)
        attention2 = F.softmax(attention2, 1).unsqueeze(2)
        mlp_input1 = torch.bmm(torch.transpose(span1_projected, 1, 2), attention1).squeeze(2)
        mlp_input2 = torch.bmm(torch.transpose(span2_projected, 1, 2), attention2).squeeze(2)
        mlp_input = torch.cat((mlp_input1, mlp_input2), dim=1)
        # mlp_input has shape (N, 512).
        mlp_output = self.sigmoid(self.linear1(mlp_input))
        mlp_output = self.sigmoid(self.linear2(mlp_output))
        mlp_output = self.sigmoid(self.output(mlp_output))

        return mlp_output


class JiantDatasetSingleSpan(data.Dataset):

    def __init__(self, encodings) -> None:
        super().__init__()
        self.encodings = encodings

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.encodings.input_ids[index]
        span1_mask = self.encodings.span1_masks[index]
        label = self.encodings.labels[index]
        return tokens, span1_mask, label

    def __len__(self):
        return len(self.encodings.input_ids)


class JiantDatasetTwoSpan(data.Dataset):

    def __init__(self, encodings) -> None:
        super().__init__()
        self.encodings = encodings

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.encodings.input_ids[index]
        span1_mask = self.encodings.span1_masks[index]
        span2_mask = self.encodings.span2_masks[index]
        label = self.encodings.labels[index]
        return tokens, span1_mask, span2_mask, label

    def __len__(self):
        return len(self.encodings.input_ids)
