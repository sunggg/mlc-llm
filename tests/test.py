from cohere.models import base_medium as cohere_medium
from utils import get_tokenizer
import torch
from transformers import GenerationConfig, GPT2Config, AutoTokenizer, GPT2LMHeadModel

tokenizer = get_tokenizer("cohere-gpt-medium")
tokenizer.pad_token_id = tokenizer.eos_token_id
torch_device = "cuda"

model = cohere_medium(
    dev=torch_device,
    dtype=torch.float16,
)

state_dict = model.state_dict()
torch.save(state_dict, "medium/pytorch_model.bin")
print("model saves to medium/pytorch_model.bin")
