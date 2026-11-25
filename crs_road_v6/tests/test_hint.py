import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["x"]
import torch
import tests.test_mock as tm
from crs import vlm


class Forced(torch.nn.Module):
    def __init__(self, tgt):
        super().__init__()
        self.tgt = tgt

    def forward(self, x):
        b = torch.tensor(vlm.xyxy_to_cxcywh(torch.tensor(self.tgt["box"])).tolist())
        cam = torch.zeros(3)
        cam[vlm.CAMS.index(self.tgt["cam"])] = 5
        kind = torch.zeros(3)
        kind[vlm.KINDS.index(self.tgt["kind_z"])] = 5
        return b, cam, kind


ds = vlm.QADataset("/tmp/syn/qa/qa_train.jsonl", "/tmp/syn/scenes")
ctx = tm.make_ctx("p3", nf=2)
ctx.bank.prepare_train()
seen = {"light": 0, "line": 0}
for i in range(len(ds)):
    q = ds.rows[i]
    if q["target"]["kind_z"] == "none":
        continue
    it = vlm.get_item(ds, i, 0, True, ctx.args)
    ctx.heads = Forced(q["target"])
    L = vlm.process_sample(ctx, it, True)
    assert "cnn" in L, q["template"]
    L["cnn"].backward()
    seen[q["target"]["kind_z"]] += 1
    if sum(seen.values()) > 12:
        break
print(seen)
ids, cl, text = vlm.zoom_hint(ctx, {"tgt": q["target"], "paths": it["paths"], "scene_id": it["scene_id"], "q": q}, torch.randn(tm.H), False, scales=(1.15, 1.3, 1.5))
print(text, cl)
