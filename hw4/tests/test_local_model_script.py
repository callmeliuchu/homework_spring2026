import os
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local optional deps
    torch = None

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if torch is not None:
    from scripts.test_local_model import build_messages, generate_completion
else:  # pragma: no cover - depends on local optional deps
    build_messages = None
    generate_completion = None


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ):
        self.last_messages = messages
        return torch.tensor([[11, 12]], dtype=torch.long)

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(token)) for token in token_ids.tolist())


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return torch.tensor([[11, 12, 99, 100]], dtype=torch.long)


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestLocalModelScriptTest(unittest.TestCase):
    def test_build_messages_wraps_prompt_as_chat(self):
        messages = build_messages("What is 2+2?")

        self.assertEqual(
            messages,
            [{"role": "user", "content": "What is 2+2?"}],
        )

    def test_generate_completion_decodes_only_new_tokens(self):
        model = FakeModel()
        tokenizer = FakeTokenizer()

        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt="Test prompt",
            device=torch.device("cpu"),
            max_new_tokens=8,
        )

        self.assertEqual(completion, "99 100")
        self.assertEqual(tokenizer.last_messages[0]["content"], "Test prompt")
        self.assertEqual(model.last_kwargs["max_new_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
