import json

from torch.utils.data import DataLoader


def test_sdft_dataloader_identity_collate(tmp_path):
    """Regression test: SDFTJsonlDataset yields dataclass objects.

    PyTorch's default collate_fn cannot collate arbitrary dataclass instances.
    We therefore must use `identity_collate` in the SDFT training loop.
    """
    from llm_mhc_sdft_tttd.data.sdft_dataset import SDFTJsonlDataset, identity_collate

    p = tmp_path / "demo.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "hi", "demonstration": "hello"}) + "\n")
        f.write(json.dumps({"prompt": "bye", "demonstration": "goodbye"}) + "\n")

    ds = SDFTJsonlDataset(str(p))
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=identity_collate)

    batch = next(iter(dl))
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert hasattr(batch[0], "prompt")
    assert hasattr(batch[0], "demonstration")


def test_sdft_dataset_accepts_utf8_bom(tmp_path):
    from llm_mhc_sdft_tttd.data.sdft_dataset import SDFTJsonlDataset

    p = tmp_path / "demo_bom.jsonl"
    # Prefix a UTF-8 BOM explicitly.
    bom = b"\xef\xbb\xbf"
    line = b'{"prompt":"hi","demonstration":"hello"}\n'
    p.write_bytes(bom + line)

    ds = SDFTJsonlDataset(str(p))
    assert len(ds) == 1
    assert ds[0].prompt == "hi"
