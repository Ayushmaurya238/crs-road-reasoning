import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crs import cnn as C
from crs import vlm

V, H = 200, 16


class Tok:
    def encode(self, t, add_special_tokens=False):
        return [ord(c) % 50 + 10 for c in t]

    def convert_tokens_to_ids(self, t):
        return 5


class Proc:
    tokenizer = Tok()

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return msgs[0]["content"][-1]["text"]

    def __call__(self, text, images, return_tensors):
        ids = [1, 2, 3] + Tok().encode(text[0])[:30] + [5, 6, 7, 8]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "pixel_values": torch.randn(4, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }


class Out:
    pass


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.lora_A = nn.Linear(H, H)
        self.head = nn.Linear(H, V)
        self.emb.weight.requires_grad = False
        self.head.weight.requires_grad = False

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, output_hidden_states=False, use_cache=False):
        h = self.emb(input_ids)
        h = h + self.lora_A(h)
        o = Out()
        o.logits = self.head(h)
        o.hidden_states = (h,)
        return o


def make_ctx(variant, nf=1):
    args = types.SimpleNamespace(
        variant=variant, lora="all", seed=0, cot_frac=0.5, img_side=64, blank=False, link_weighted=True,
        crop_size=64, hint_flip=0.15, tta=True, oracle_box=False, lr=1e-3, lora_plus=8.0, dtype="bf16",
    )
    ctx = vlm.Ctx()
    ctx.args, ctx.dev, ctx.dtype = args, torch.device("cpu"), torch.float32
    ctx.proc, ctx.model, ctx.hs = Proc(), Model(), H
    vlm.finish_ctx(ctx)
    vlm.attach_heads(ctx, args)
    if variant == "p3":
        ctx.bank = C.CnnBank("googlenet", H, nf, ctx.dev)
    return ctx


def main():
    ds = vlm.QADataset("/tmp/syn/qa/qa_train.jsonl", "/tmp/syn/scenes", 40)
    ctx = make_ctx("p2")
    for i in range(len(ds)):
        it = vlm.get_item(ds, i, 0, True, ctx.args)
        L = vlm.process_sample(ctx, it, False)
        t = vlm.total_loss(L, vlm.stage_weights("B", {"rank": 1.0, "box": 1.0, "aux": 1.0}, True))
        t.backward()
    cal = {"rank": 1.0, "box": 1.0, "aux": 1.0}
    print("calib", vlm.calibrate(ctx, vlm.get_item(ds, 0, 0, True, ctx.args), cal), cal)
    ctx3 = make_ctx("p3", nf=2)
    ctx3.bank.prepare_train()
    copt = ctx3.bank.optimizer()
    n_cnn = 0
    for i in range(len(ds)):
        it = vlm.get_item(ds, i, 0, True, ctx3.args)
        L = vlm.process_sample(ctx3, it, True)
        n_cnn += "cnn" in L
        vlm.total_loss(L, vlm.stage_weights("C", cal, True)).backward()
    copt.step()
    print("samples with cnn loss", n_cnn)
    ctx1 = make_ctx("p1")
    ctx1.heads = None
    it = vlm.get_item(ds, 0, 0, True, ctx1.args)
    L = vlm.process_sample(ctx1, it, False)
    vlm.total_loss(L, vlm.stage_weights("A", cal, False)).backward()
    dv = vlm.QADataset("/tmp/syn/qa/qa_dev.jsonl", "/tmp/syn/scenes", 12)
    print(vlm.quick_val(ctx, dv, 6))
    for c, name in ((ctx, "p2"), (ctx3, "p3")):
        r = vlm.evaluate(c, dv, c.args, f"/tmp/preds_{name}.jsonl")
        print(name, r)
    ctx3.args.oracle_box = True
    print("oracle", vlm.evaluate(ctx3, dv, ctx3.args, "/tmp/preds_orc.jsonl"))
    ctx.args.lora = "none"
    print("save"); 
    for lr in (0, 5, 10, 14, 19):
        print(lr, round(vlm.lr_mult(lr, 20, 14, True), 3))


if __name__ == "__main__":
    main()
