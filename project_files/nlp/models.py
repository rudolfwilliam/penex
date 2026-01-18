"""PENEX required rescaling of the logits during evaluation, which is why we need a custom model class to handle this."""

import copy

from transformers import RobertaForSequenceClassification
import torch


class PENEXLMHeadWrapper(torch.nn.Module):
    """
        A wrapper for the final output layer that rescales logits during evaluation.
    """
    def __init__(self, lm_head, sensitivity):
        super().__init__()
        self.lm_head = lm_head # linear layer
        self.sensitivity_ = sensitivity
    
    @property
    def sensitivity(self):
        return self.sensitivity_
    
    @sensitivity.setter
    def sensitivity(self, value):
        self.sensitivity_ = value

    def forward(self, hidden_states, **kwargs):
        # Call the original forward
        logits = self.lm_head(hidden_states, **kwargs)

        # If we're in evaluation mode, rescale the logits
        if not self.training:
            logits *= (1 + self.sensitivity)

        return logits


class PENEXRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, sensitivity):
        super().__init__(config)
        # Wrapper to rescale the final logits during evaluation using custom lm_head
        self.classifier.out_proj = PENEXLMHeadWrapper(copy.deepcopy(self.classifier.out_proj), sensitivity)
    
    @property
    def sensitivity(self):
        return self.classifier.out_proj.sensitivity
    
    @sensitivity.setter
    def sensitivity(self, value):
        self.classifier.out_proj.sensitivity = value
