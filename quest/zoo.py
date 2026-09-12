from __future__ import annotations

from pathlib import Path

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

# Row order of every marker table shipped here. The files carry their tables as bare tensors, so
# this list IS the mapping from a marker name to a row; reordering it silently repoints every
# query in every model.
QUEST_MARKERS = [
    "DAPI", "CD45", "CD31", "CD4", "CD11c", "HLA-DR", "CD68", "CD3e", "CD8", "Ki67", "FoxP3",
    "PD1", "CD14", "PDL1", "CD20", "PanCK", "Podoplanin", "CD45RO", "Vimentin", "aSMA", "CD38",
    "CD21", "GranzymeB", "CD163", "CD141", "CD44", "Gal3", "EpCAM", "ICOS", "BCL2", "LAG3",
    "CD34", "CollagenIV", "VISTA", "ECad", "IDO1", "PCNA", "GATA3", "IFNg", "Keratin8/18",
    "HLA-ABC", "TP63", "MPO", "CD66", "CD40", "HLA-E", "CD79", "Caveolin1", "CD39", "CD11b",
    "CD45RA", "PGP9.5", "ATM", "CD16", "CD56", "TIGIT", "ERa", "CD19", "CD117", "CD15",
    "CD107a", "CD134", "CD197", "CD183", "CD25", "TIM3", "CD127", "TCF1", "CD208", "CD137",
    "SOX10", "CD49", "CD57", "CD69", "CD47", "p16", "TMEM16A", "CD152", "Siglec8", "CD1c",
    "LYVE1", "Perforin", "CD66b", "CX3CR1", "bCatenin", "Tbet", "TOX", "Nestin", "Olig2",
    "CD196", "XCR1", "CD206", "TFAM", "HistoneH3p", "Keratin14", "CD194", "RORgammaT",
    "Keratin19", "CD138", "CD62L", "F4/80", "Tryptase", "S100A4", "CD207", "CXCR5", "CXCL13",
    "FAP", "BCL6", "CD209", "PNAD", "INOS", "pSTAT3", "EGFR", "p53", "TCRgammadelta", "CD123",
    "CD90", "CD227", "CD27", "CD33", "Perlecan", "Clusterin", "CD140b", "PDL2", "CD103",
    "TCRb", "Ly6G", "CD5", "CD71"
]

_QUEST_ARCH = {
    "image_size": 224, "patch_size": 14,
    "he_encoder": "cached_uni2", "he_encoder_dim": 1536,
    "decoder_dim": 512, "pixel_dim": 256,
    "num_cross_attn_layers": 8, "num_heads": 8, "dropout": 0.0,
    "use_film": True, "keep_marker_residual": True, "marker_self_attention": False,
    "use_marker_conv_head": False, "use_he_image_skip": False,
    "use_he_pyramid_decoder": True, "he_pyramid_backbone": "convnext_base",
    "he_pyramid_backbone_pretrained": True, "he_pyramid_backbone_trainable": False,
    "he_pyramid_backbone_out_indices": [0, 1, 2, 3],
    "use_marker_presence_head": False, "mask_queries_per_marker": 1,
    "output_activation": "none", "predict_uncertainty": False,
    "pixel_hidden_dims": [512, 384, 256],
    "marker_embed": "mistral3",
    "marker_center": False, "marker_normalize": False, "marker_identity": False,
}

QUEST_SEMANTIC_CONFIG = {**_QUEST_ARCH, "marker_dim": 3072}
QUEST_ID_CONFIG = {**_QUEST_ARCH, "marker_dim": 4096}
QUEST_EVA_HE_MARKERS = ["HECHA1", "HECHA2", "HECHA3"] 
QUEST_EVA_MARKERS = list(QUEST_MARKERS) + QUEST_EVA_HE_MARKERS
QUEST_EVA_CONFIG = {
    "ds": {"patch_size": 224, "token_size": 8, "marker_dim": 3072,
           "mask_strategy": "random", "mask_ratio": 0.0},
    "cm": {"dim": 512, "mlp_ratio": 4, "n_heads": 4, "n_layers": 2},
    "pm": {"dim": 768, "mlp_ratio": 4, "n_heads": 12, "n_layers": 12, "out_dim": 512},
    "de": {"dim": 512, "marker_dim": 512, "mlp_ratio": 4, "n_heads": 16, "n_layers": 8},
}

MODELS = {
    "quest-semantic": {"ckpt": "quest_semantic.ckpt", "family": "quest",
                       "config": QUEST_SEMANTIC_CONFIG, "markers": QUEST_MARKERS},
    "quest-id": {"ckpt": "quest_id.ckpt", "family": "quest",
                 "config": QUEST_ID_CONFIG, "markers": QUEST_MARKERS},
    "quest-eva": {"ckpt": "quest_eva.ckpt", "family": "eva",
                  "config": QUEST_EVA_CONFIG, "markers": QUEST_EVA_MARKERS},
}
