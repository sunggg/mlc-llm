from .model import HeadModel

from tokenizers import Tokenizer
import torch
from transformers import GenerationConfig, GPT2Config, AutoTokenizer, GPT2LMHeadModel


def base_medium(dev="cpu", dtype=torch.float16):
    config = GPT2Config(
        vocab_size=51200,
        n_positions=2048,
        n_embd=4096,
        n_layer=28,
        n_head=32,
        n_inner=16384,
        activation_function="gelu",
    )
    return HeadModel(config, device=dev, dtype=dtype)


def base_xlarge(dev="cpu", dtype=torch.float16):
    config = GPT2Config(
        vocab_size=51200,
        n_positions=2048,
        n_embd=8192,
        n_layer=64,
        n_head=64,
        n_inner=32768,
        activation_function="gelu",
    )
    return HeadModel(config, device=dev, dtype=dtype)


if __name__ == "__main__":
    # Make the model deterministic
    torch.manual_seed(0)
    dev = "cuda:0"
    model = base_medium(dev=dev, dtype=torch.float16)
    model.eval()

    # uncomment to print model architecture
    # print(model)

    # Cohere did not provide us with their tokenizer json file, so use gpt2 tokenizer instead.
    # tokenizer = Tokenizer.from_file("data/cohere.json")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    prompt = "My name is Teven and I am"
    output = tokenizer.encode(prompt)

    tokens = torch.IntTensor(tokenizer.encode(prompt)).unsqueeze(0).to(dev)
    completion = model.generate(
        input_ids=tokens,
        generation_config=GenerationConfig(
            top_k=1,
            temperature=0.0,
        ),
        pad_token_id=0,
        # TODO eos_token_id=tokenizer.token_to_id
        max_new_tokens=100,
    )
    print(tokenizer.decode(completion.squeeze().tolist()))
