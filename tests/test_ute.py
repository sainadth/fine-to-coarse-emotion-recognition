import torch

from fine2coarse.models.ute import UtteranceLevelTransformerEncoder


def test_output_shape_matches_input():
    ute = UtteranceLevelTransformerEncoder(d_model=32, num_layers=1, num_heads=4, dim_feedforward=64)
    hidden = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert ute(hidden, mask).shape == (2, 5, 32)


def test_padded_turns_do_not_change_real_turns():
    torch.manual_seed(0)
    ute = UtteranceLevelTransformerEncoder(d_model=32, num_layers=1, num_heads=4, dim_feedforward=64).eval()
    real = torch.randn(1, 3, 32)

    short_mask = torch.ones(1, 3, dtype=torch.bool)
    padded = torch.cat([real, torch.randn(1, 2, 32)], dim=1)
    padded_mask = torch.tensor([[True, True, True, False, False]])

    with torch.no_grad():
        out_short = ute(real, short_mask)
        out_padded = ute(padded, padded_mask)

    assert torch.allclose(out_short, out_padded[:, :3], atol=1e-5)


def test_position_embeddings_make_order_matter():
    torch.manual_seed(0)
    ute = UtteranceLevelTransformerEncoder(d_model=32, num_layers=1, num_heads=4, dim_feedforward=64).eval()
    hidden = torch.randn(1, 4, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        forward = ute(hidden, mask)
        reversed_ = ute(hidden.flip(1), mask)
    assert not torch.allclose(forward.flip(1), reversed_, atol=1e-4)


def test_speaker_embedding_is_initialised_like_the_positional_one():
    """Both are added straight onto RoBERTa hidden states, so they must start at the same
    scale. nn.Embedding defaults to N(0, 1), which is ~50x the positional std of 0.02 and
    swamps the representation at initialisation."""
    encoder = UtteranceLevelTransformerEncoder(d_model=64, num_speakers=9)
    assert encoder.speaker is not None
    assert encoder.speaker.weight.std().item() < 0.05
    assert abs(encoder.speaker.weight.std().item() - encoder.position.weight.std().item()) < 0.02
