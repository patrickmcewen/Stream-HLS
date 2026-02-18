import torch
from utils import randTensor

dtype = torch.float32

# fix torch seed
torch.manual_seed(0)
# synthetic params
I, J, K, L, M, N, O, P, Q = 64, 32, 64, 64, 128, 128, 64, 32, 16
S = 64

model_configs = {
  # new benchmark
  "new_benchmark" : {
    "MNIST" : {
      "class": "MNIST",
      "config" : {},
      "input" : (
        randTensor(8, 1024, dtype=dtype),
      )
    },
  },
  # polybench medium
  "polybench" : {
    "gemm" : {
      "class": "gemm",
      "config" : {},
      "input" : (
        randTensor(200, 240, dtype=dtype),
        randTensor(240, 220, dtype=dtype),
        randTensor(200, 220, dtype=dtype)
      )
    },
    "k2mm" : {
      "class": "k2mm",
      "config" : {},
      "input" : (
        randTensor(180, 210, dtype=dtype),     # A
        randTensor(210, 190, dtype=dtype),     # B
        randTensor(190, 220, dtype=dtype),     # C
        randTensor(180, 220, dtype=dtype)      # D
      )
    },
    "k3mm" : {
      "class": "k3mm",
      "config" : {},
      "input" : (
        randTensor(180, 180, dtype=dtype),
        randTensor(180, 180, dtype=dtype),
        randTensor(180, 180, dtype=dtype),
        randTensor(180, 180, dtype=dtype)
      )
    },
    "k3mm_call" : {
      "class": "k3mm_call",
      "config" : {},
      "input" : (
        randTensor(180, 200, dtype=dtype),
        randTensor(200, 190, dtype=dtype),
        randTensor(190, 220, dtype=dtype),
        randTensor(220, 210, dtype=dtype)
      )
    },
    "atax" : {
      "class": "atax",
      "config" : {},
      "input" : (
        randTensor(390, 410, dtype=dtype),  # A
        randTensor(410, dtype=dtype)        # x
      )
    },
    "bicg" : {
      "class": "bicg",
      "config" : {},
      "input" : (
        randTensor(410, 390, dtype=dtype),  # A
        randTensor(410, 390, dtype=dtype),  # A copy
        randTensor(410, dtype=dtype),       # r
        randTensor(390, dtype=dtype)        # p
      )
    },
    "kernel_bicg" : {
      "class": "bicg",
      "config" : {},
      "input" : (
        randTensor(410, 390, dtype=dtype),  # A
        randTensor(410, 390, dtype=dtype),  # A copy
        randTensor(410, dtype=dtype),       # r
        randTensor(390, dtype=dtype)        # p
      )
    },
    "mvt" : {
      "class": "mvt",
      "config" : {},
      "input" : (
        randTensor(400, 400, dtype=dtype),  # A
        randTensor(400, 400, dtype=dtype),  # A copy
        randTensor(400, dtype=dtype),       # y1
        randTensor(400, dtype=dtype)        # y2
      )
    },
    "gesummv" : {
      "class": "gesummv",
      "config" : {},
      "input" : (
        randTensor(250, 250, dtype=dtype),  # A
        randTensor(250, 250, dtype=dtype),  # B
        randTensor(250, dtype=dtype)       # x
      )
    },
    "resnet18" : {
      "class": "resnet18",
      "config" : {},
      "input" : (
        randTensor(1, 3, 224, 224, dtype=dtype),
      )
    }
  },
  # mlps
  "mlps" : {
    "Autoencoder" : {
      "class": "Autoencoder",
      "config" : {},
      "input" : (
        randTensor(32, 784, dtype=dtype),
      )
    },
    "ResMLP" : {
      "class": "ResMLP",
      "config" : {},
      "input" : (
        randTensor(8, 1024, dtype=dtype),
      )
    },
    "Decoder" : {
      "class": "Decoder",
      "config" : dict(
        input_size=1024
      ),
      "input" : (
        randTensor(1, 1024, dtype=dtype),
      )
    },
    "DenseDecoder" : {
      "class": "DenseDecoder",
      "config" : dict(
        input_size=474
      ),
      "input" : (
        randTensor(2, 474, dtype=dtype),
      )
    }
  },
  # cnn
  "cnns" : {
    "DepthwiseSeparableConvBlock" : {
      "class": "DepthwiseSeparableConvBlock",
      "config" : dict(
        in_channels=8,
        out_channels=8,
        stride=1
      ),
      "input" : (
        randTensor(1, 8, 112, 112, dtype=dtype),
      )
    },
    "ResidualBlock" : {
      "class": "ResidualBlock",
      "config" : dict(
        in_channels=16,
        out_channels=16,
        stride=1
      ),
      "input" : (
        randTensor(1, 16, 56, 56, dtype=dtype),
      )
    },
  },
  # transformers
  "transformers" : {
    "FeedForward" : {
      "class": "FeedForward",
      "config" : dict(
        embed_dim=128,
        ff_dim=256
      ),
      "input" : (
        randTensor(1, 512, 128, dtype=dtype),
      )
    },
    "MultiHeadSelfAttention" : {
      "class": "MultiHeadSelfAttention",
      "config" : dict(
        embed_dim=128,
        num_heads=8
      ),
      "input" : (
        randTensor(1, 64, 128, dtype=dtype),
      )
    },
    "MultiHeadSelfAttentionKV" : {
      "class": "MultiHeadSelfAttentionKV",
      "config" : dict(
        embed_dim=128,
        num_heads=8
      ),
      "input" : (
        randTensor(1, 1, 128, dtype=dtype),  # x: single new token
        randTensor(1, 1024, 8, 16, dtype=dtype),  # k_cache: cached keys (batch, cached_len, num_heads, head_dim)
        randTensor(1, 1024, 8, 16, dtype=dtype),  # v_cache: cached values (batch, cached_len, num_heads, head_dim)
      )
    },
    "bit_llama" : {
      "class": "bit_llama",
      "config" : {},
      "input" : (
        torch.randint(0, 20000, (1, 512), dtype=torch.long),
      )
    },
    "bit_transformer" : {
      "class": "bit_transformer",
      "config" : dict(
        dim=128,
        depth=6,
        num_tokens=20000,
        heads=8,
        ff_mult=4
      ),
      "input" : (
        # Provide embedded tensors directly (batch, seq_len, dim) instead of token indices
        # to avoid unsupported MLIR operations (embedding lookup done on host side)
        randTensor(1, 512, 128, dtype=dtype),
      )
    }
  },
  "LLM" : {
    "Transformer" : {
      "class": "Transformer",
      "config" : dict(
        embed_dim=128,
        num_heads=8,
        num_layers=6,
        num_tokens=20000,
        ff_dim=256
      ),
      "input" : (
        torch.randint(0, 20000, (1, 512), dtype=torch.long),
      )
    },
    "TransformerKV" : {
      "class": "TransformerKV",
      "config" : dict(
        embed_dim=128,
        num_heads=4,
        num_layers=1,
        num_tokens=20000,
        ff_dim=256
      ),
      "input" : (
        #torch.randint(0, 20000, (1, 1), dtype=torch.long),
        randTensor(1, 1, 128, dtype=dtype), 
        randTensor(1, 1024, 4, 32, dtype=dtype),
        randTensor(1, 1024, 4, 32, dtype=dtype),
      )
    },
    "Inference" : {
      "class": "Inference",
      "config" : dict(
        embed_dim=128,
        num_heads=4,
        num_layers=1,
        num_tokens=20000,
        ff_dim=256
      ),
      "input" : (
        #torch.randint(0, 20000, (1, 512), dtype=torch.long),
        randTensor(1, 512, 128, dtype=dtype),
      )
    }
  },
  "test_bitnet" : {
    "BitLinear" : {
      "class": "BitLinear",
      "config" : dict(
        in_features=128,
        out_features=128
      ),
      "input" : (
        randTensor(1, 128, 128, dtype=dtype),
      )
    },
    "SimpleRMSNorm" : {
      "class": "SimpleRMSNorm",
      "config" : dict(
        dim=128
      ),
      "input" : (
        randTensor(1, 128, 128, dtype=dtype),
      )
    },
    "BitFeedForward" : {
      "class": "BitFeedForward",
      "config" : dict(
        dim=128
      ),
      "input" : (
        randTensor(1, 128, 128, dtype=dtype),
      )
    },
    "BitMGQA" : {
      "class": "BitMGQA",
      "config" : dict(
        embed_dim=8,
        query_heads=1,
        kv_heads=1
      ),
      "input" : (
        randTensor(1, 512, 8, dtype=dtype),
        randTensor(1, 512, 8, dtype=dtype),
        randTensor(1, 512, 8, dtype=dtype),
      )
    },
    "BitNetTransformer" : {
      "class": "BitNetTransformer",
      "config" : dict(
        dim=32,
        depth=1,
        num_tokens=20000,
      ),
      "input" : (
        torch.randint(0, 20000, (1, 1024)),
      )
    },
    "BitNetInference" : {
      "class": "BitNetInference",
      "config" : {},
      "input" : (
        #torch.randint(0, 20000, (1, 1024)),
        randTensor(1, 1024, 32, dtype=dtype),
      )
    }
  },
  "test_llama" : {
    "RMSNorm" : {
      "class": "RMSNorm",
      "config" : dict(
        dim=128,
        norm_eps=1e-5
      ),
      "input" : (
        randTensor(1, 128, 128, dtype=dtype),
      )
    },
    "FeedForward" : {
      "class": "FeedForward",
      "config" : dict(
      ),
      "input" : (
        randTensor(1, 14336, 4096, dtype=dtype),
      )
    },
    "Attention" : {
      "class": "Attention",
      "config" : dict(
      ),
      "input" : (
        randTensor(2, 8, 4096, dtype=dtype),
        randTensor(8, 64, dtype=dtype),
        randTensor(8, 8, dtype=dtype),
        randTensor(4, 128, 8, 128, dtype=dtype),
        randTensor(4, 128, 8, 128, dtype=dtype),
      )
    }
  },
  # synthetic
  "synthetic" : {
    "k7mmseq_balanced" : {
      "class": "k7mmseq",
      "config" : {},
      "input" : (
        randTensor(S, S, dtype=dtype),
        randTensor(S, S, dtype=dtype),
        randTensor(S, S, dtype=dtype),
        randTensor(S, S, dtype=dtype),
        randTensor(S, S, dtype=dtype),
        randTensor(S, S, dtype=dtype),
        randTensor(S, S, dtype=dtype),
        randTensor(S, S, dtype=dtype)
      )
    },
    "k7mmseq_unbalanced" : {
      "class": "k7mmseq",
      "config" : {},
      "input" : (
        randTensor(I, J, dtype=dtype),
        randTensor(J, K, dtype=dtype),
        randTensor(K, L, dtype=dtype),
        randTensor(L, M, dtype=dtype),
        randTensor(M, N, dtype=dtype),
        randTensor(N, O, dtype=dtype),
        randTensor(O, P, dtype=dtype),
        randTensor(P, Q, dtype=dtype)
      )
    }
  },
  "codesign" : {
    "MultiHeadSelfAttention" : {
      "class": "MultiHeadSelfAttention",
      "config" : dict(
        embed_dim=128,
        num_heads=8
      ),
      "input" : (
        randTensor(1, 64, 128, dtype=dtype),
      )
    },
    "FeedForward" : {
      "class": "FeedForward",
      "config" : dict(
        embed_dim=128,
        ff_dim=256
      ),
      "input" : (
        randTensor(1, 512, 128, dtype=dtype),
      )
    },
    "LlamaAttention" : {
      "class": "LlamaAttention",
      "config" : dict(
        dim=256,
        n_heads=4
      ),
      "input" : (
        randTensor(1, 64, 256, dtype=dtype),
      )
    },
    "LlamaFeedForward" : {
      "class": "LlamaFeedForward",
      "config" : dict(
        dim=256,
        ffn_dim=1024
      ),
      "input" : (
        randTensor(1, 64, 256, dtype=dtype),
      )
    },
    "LlamaPrefillAttention" : {
      "class": "LlamaPrefillAttention",
      "config" : dict(
        dim=256,
        n_heads=4
      ),
      "input" : (
        randTensor(1, 64, 256, dtype=dtype),
      )
    },
    "LlamaDecodeAttention" : {
      "class": "LlamaDecodeAttention",
      "config" : dict(
        dim=256,
        n_heads=4,
        cache_len=64
      ),
      "input" : (
        randTensor(1, 1, 256, dtype=dtype),            # x: single token
        randTensor(1, 4, 64, 64, dtype=dtype),          # k_cache
        randTensor(1, 4, 64, 64, dtype=dtype),          # v_cache
      )
    },
    "LlamaDecodeFeedForward" : {
      "class": "LlamaDecodeFeedForward",
      "config" : dict(
        dim=256,
        ffn_dim=1024
      ),
      "input" : (
        randTensor(1, 1, 256, dtype=dtype),
      )
    },
    # ---------------------------------------------------------------------------
    # Mixtral 8x7B sub-blocks (hidden=4096, GQA 32/8 heads, top-2 MoE)
    # ---------------------------------------------------------------------------
    "MixtralPrefillAttention" : {
      "class": "MixtralPrefillAttention",
      "config" : dict(
        hidden=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128
      ),
      "input" : (
        randTensor(1, 512, 4096, dtype=dtype),
      )
    },
    "MixtralPrefillMoE" : {
      "class": "MixtralPrefillMoE",
      "config" : dict(
        hidden=4096,
        ffn_intermediate=14336,
        n_experts=8,
        top_k=2
      ),
      "input" : (
        randTensor(1, 512, 4096, dtype=dtype),
      )
    },
    "MixtralDecodeAttention" : {
      "class": "MixtralDecodeAttention",
      "config" : dict(
        hidden=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        cache_len=512
      ),
      "input" : (
        randTensor(1, 1, 4096, dtype=dtype),           # x: new token
        randTensor(1, 8, 512, 128, dtype=dtype),        # k_cache
        randTensor(1, 8, 512, 128, dtype=dtype),        # v_cache
      )
    },
    "MixtralDecodeMoE" : {
      "class": "MixtralDecodeMoE",
      "config" : dict(
        hidden=4096,
        ffn_intermediate=14336,
        n_experts=8,
        top_k=2
      ),
      "input" : (
        randTensor(1, 1, 4096, dtype=dtype),
      )
    },
    # ---------------------------------------------------------------------------
    # DeepSeek-V3 sub-blocks (hidden=7168, MLA attention, dense + MoE FFN)
    # ---------------------------------------------------------------------------
    "DeepSeekPrefillAttention" : {
      "class": "DeepSeekPrefillAttention",
      "config" : dict(
        hidden=7168,
        n_heads=128,
        head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512
      ),
      "input" : (
        randTensor(1, 512, 7168, dtype=dtype),
      )
    },
    "DeepSeekPrefillDenseFFN" : {
      "class": "DeepSeekPrefillDenseFFN",
      "config" : dict(
        hidden=7168,
        ffn_intermediate=18432
      ),
      "input" : (
        randTensor(1, 512, 7168, dtype=dtype),
      )
    },
    "DeepSeekPrefillMoE" : {
      "class": "DeepSeekPrefillMoE",
      "config" : dict(
        hidden=7168,
        moe_intermediate=2048,
        n_routed_experts=256,
        top_k=8
      ),
      "input" : (
        randTensor(1, 512, 7168, dtype=dtype),
      )
    },
    "DeepSeekDecodeAttention" : {
      "class": "DeepSeekDecodeAttention",
      "config" : dict(
        hidden=7168,
        n_heads=128,
        head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        cache_len=512
      ),
      "input" : (
        randTensor(1, 1, 7168, dtype=dtype),              # x: new token
        randTensor(1, 128, 512, 128, dtype=dtype),         # k_cache
        randTensor(1, 128, 512, 128, dtype=dtype),         # v_cache
      )
    },
    "DeepSeekDecodeDenseFFN" : {
      "class": "DeepSeekDecodeDenseFFN",
      "config" : dict(
        hidden=7168,
        ffn_intermediate=18432
      ),
      "input" : (
        randTensor(1, 1, 7168, dtype=dtype),
      )
    },
    "DeepSeekDecodeMoE" : {
      "class": "DeepSeekDecodeMoE",
      "config" : dict(
        hidden=7168,
        moe_intermediate=2048,
        n_routed_experts=256,
        top_k=8
      ),
      "input" : (
        randTensor(1, 1, 7168, dtype=dtype),
      )
    },
    # ---------------------------------------------------------------------------
    # V-JEPA sub-blocks (ViT-H/16 encoder + narrow predictor)
    # ---------------------------------------------------------------------------
    "VJEPASpatialAttention" : {
      "class": "VJEPASpatialAttention",
      "config" : dict(
        hidden=1280,
        n_heads=16,
        head_dim=80
      ),
      "input" : (
        randTensor(16, 196, 1280, dtype=dtype),   # (n_frames, spatial_patches, hidden)
      )
    },
    "VJEPATemporalAttention" : {
      "class": "VJEPATemporalAttention",
      "config" : dict(
        hidden=1280,
        n_heads=16,
        head_dim=80
      ),
      "input" : (
        randTensor(1, 3136, 1280, dtype=dtype),   # (batch, total_patches, hidden)
      )
    },
    "VJEPAFeedForward" : {
      "class": "VJEPAFeedForward",
      "config" : dict(
        hidden=1280,
        ffn_intermediate=5120
      ),
      "input" : (
        randTensor(1, 3136, 1280, dtype=dtype),
      )
    },
    "VJEPAPredictorAttention" : {
      "class": "VJEPAPredictorAttention",
      "config" : dict(
        pred_hidden=384,
        n_heads=12,
        head_dim=32
      ),
      "input" : (
        randTensor(1, 3136, 384, dtype=dtype),    # (batch, pred_seq_len, pred_hidden)
      )
    },
    "VJEPAPredictorFeedForward" : {
      "class": "VJEPAPredictorFeedForward",
      "config" : dict(
        pred_hidden=384,
        ffn_intermediate=1536
      ),
      "input" : (
        randTensor(1, 3136, 384, dtype=dtype),
      )
    },
    # ---------------------------------------------------------------------------
    # OpenVLA sub-blocks (DINOv2 + SigLIP + vision projection + Llama2-7B LM)
    # ---------------------------------------------------------------------------
    "OpenVLADinoAttention" : {
      "class": "OpenVLADinoAttention",
      "config" : dict(
        hidden=1024,
        n_heads=16,
        head_dim=64
      ),
      "input" : (
        randTensor(1, 256, 1024, dtype=dtype),    # (batch, n_patches, dino_hidden)
      )
    },
    "OpenVLADinoFeedForward" : {
      "class": "OpenVLADinoFeedForward",
      "config" : dict(
        hidden=1024,
        ffn_intermediate=4096
      ),
      "input" : (
        randTensor(1, 256, 1024, dtype=dtype),
      )
    },
    "OpenVLAVisionAttention" : {
      "class": "OpenVLAVisionAttention",
      "config" : dict(
        hidden=1152,
        n_heads=16,
        head_dim=72
      ),
      "input" : (
        randTensor(1, 256, 1152, dtype=dtype),    # (batch, n_patches, vis_hidden)
      )
    },
    "OpenVLAVisionFeedForward" : {
      "class": "OpenVLAVisionFeedForward",
      "config" : dict(
        hidden=1152,
        ffn_intermediate=4304
      ),
      "input" : (
        randTensor(1, 256, 1152, dtype=dtype),
      )
    },
    "OpenVLAVisionProjection" : {
      "class": "OpenVLAVisionProjection",
      "config" : dict(
        proj_in=2176,
        lm_hidden=4096
      ),
      "input" : (
        randTensor(1, 256, 2176, dtype=dtype),    # (batch, n_patches, dino+siglip concat)
      )
    },
    "OpenVLALMPrefillAttention" : {
      "class": "OpenVLALMPrefillAttention",
      "config" : dict(
        hidden=4096,
        n_heads=32,
        head_dim=128
      ),
      "input" : (
        randTensor(1, 512, 4096, dtype=dtype),    # (batch, vision+lang tokens, lm_hidden)
      )
    },
    "OpenVLALMFeedForward" : {
      "class": "OpenVLALMFeedForward",
      "config" : dict(
        hidden=4096,
        ffn_intermediate=11008
      ),
      "input" : (
        randTensor(1, 512, 4096, dtype=dtype),
      )
    },
    "OpenVLALMDecodeAttention" : {
      "class": "OpenVLALMDecodeAttention",
      "config" : dict(
        hidden=4096,
        n_heads=32,
        head_dim=128,
        cache_len=512
      ),
      "input" : (
        randTensor(1, 1, 4096, dtype=dtype),              # x: new action token
        randTensor(1, 32, 512, 128, dtype=dtype),          # k_cache
        randTensor(1, 32, 512, 128, dtype=dtype),          # v_cache
      )
    },
    # ---------------------------------------------------------------------------
    # DeepSeek-V3 SMALL sub-blocks (hidden=512, prompt_len=64, cache_len=64)
    # ---------------------------------------------------------------------------
    "DeepSeekPrefillAttentionSmall" : {
      "class": "DeepSeekPrefillAttention",
      "config" : dict(
        hidden=512,
        n_heads=8,
        head_dim=64,
        q_lora_rank=128,
        kv_lora_rank=64
      ),
      "input" : (
        randTensor(1, 64, 512, dtype=dtype),
      )
    },
    "DeepSeekPrefillDenseFFNSmall" : {
      "class": "DeepSeekPrefillDenseFFN",
      "config" : dict(
        hidden=512,
        ffn_intermediate=1024
      ),
      "input" : (
        randTensor(1, 64, 512, dtype=dtype),
      )
    },
    "DeepSeekPrefillMoESmall" : {
      "class": "DeepSeekPrefillMoE",
      "config" : dict(
        hidden=512,
        moe_intermediate=256,
        n_routed_experts=16,
        top_k=2
      ),
      "input" : (
        randTensor(1, 64, 512, dtype=dtype),
      )
    },
    "DeepSeekDecodeAttentionSmall" : {
      "class": "DeepSeekDecodeAttention",
      "config" : dict(
        hidden=512,
        n_heads=8,
        head_dim=64,
        q_lora_rank=128,
        kv_lora_rank=64,
        cache_len=64
      ),
      "input" : (
        randTensor(1, 1, 512, dtype=dtype),               # x: new token
        randTensor(1, 8, 64, 64, dtype=dtype),             # k_cache
        randTensor(1, 8, 64, 64, dtype=dtype),             # v_cache
      )
    },
    "DeepSeekDecodeDenseFFNSmall" : {
      "class": "DeepSeekDecodeDenseFFN",
      "config" : dict(
        hidden=512,
        ffn_intermediate=1024
      ),
      "input" : (
        randTensor(1, 1, 512, dtype=dtype),
      )
    },
    "DeepSeekDecodeMoESmall" : {
      "class": "DeepSeekDecodeMoE",
      "config" : dict(
        hidden=512,
        moe_intermediate=256,
        n_routed_experts=16,
        top_k=2
      ),
      "input" : (
        randTensor(1, 1, 512, dtype=dtype),
      )
    },
    # ---------------------------------------------------------------------------
    # Mixtral SMALL sub-blocks (hidden=512, n_kv_heads=2, prompt_len=64)
    # ---------------------------------------------------------------------------
    "MixtralPrefillAttentionSmall" : {
      "class": "MixtralPrefillAttention",
      "config" : dict(
        hidden=512,
        n_heads=8,
        n_kv_heads=2,
        head_dim=64
      ),
      "input" : (
        randTensor(1, 64, 512, dtype=dtype),
      )
    },
    "MixtralPrefillMoESmall" : {
      "class": "MixtralPrefillMoE",
      "config" : dict(
        hidden=512,
        ffn_intermediate=1024,
        n_experts=4,
        top_k=2
      ),
      "input" : (
        randTensor(1, 64, 512, dtype=dtype),
      )
    },
    "MixtralDecodeAttentionSmall" : {
      "class": "MixtralDecodeAttention",
      "config" : dict(
        hidden=512,
        n_heads=8,
        n_kv_heads=2,
        head_dim=64,
        cache_len=64
      ),
      "input" : (
        randTensor(1, 1, 512, dtype=dtype),               # x: new token
        randTensor(1, 2, 64, 64, dtype=dtype),             # k_cache
        randTensor(1, 2, 64, 64, dtype=dtype),             # v_cache
      )
    },
    "MixtralDecodeMoESmall" : {
      "class": "MixtralDecodeMoE",
      "config" : dict(
        hidden=512,
        ffn_intermediate=1024,
        n_experts=4,
        top_k=2
      ),
      "input" : (
        randTensor(1, 1, 512, dtype=dtype),
      )
    },
    # ---------------------------------------------------------------------------
    # V-JEPA SMALL sub-blocks (hidden=256, n_frames=4 → 784 total patches)
    # ---------------------------------------------------------------------------
    "VJEPASpatialAttentionSmall" : {
      "class": "VJEPASpatialAttention",
      "config" : dict(
        hidden=256,
        n_heads=4,
        head_dim=64
      ),
      "input" : (
        randTensor(4, 196, 256, dtype=dtype),   # (n_frames, spatial_patches, hidden)
      )
    },
    "VJEPATemporalAttentionSmall" : {
      "class": "VJEPATemporalAttention",
      "config" : dict(
        hidden=256,
        n_heads=4,
        head_dim=64
      ),
      "input" : (
        randTensor(1, 784, 256, dtype=dtype),   # (batch, n_frames*spatial_patches, hidden)
      )
    },
    "VJEPAFeedForwardSmall" : {
      "class": "VJEPAFeedForward",
      "config" : dict(
        hidden=256,
        ffn_intermediate=1024
      ),
      "input" : (
        randTensor(1, 784, 256, dtype=dtype),
      )
    },
    "VJEPAPredictorAttentionSmall" : {
      "class": "VJEPAPredictorAttention",
      "config" : dict(
        pred_hidden=128,
        n_heads=4,
        head_dim=32
      ),
      "input" : (
        randTensor(1, 784, 128, dtype=dtype),   # (batch, pred_seq_len, pred_hidden)
      )
    },
    "VJEPAPredictorFeedForwardSmall" : {
      "class": "VJEPAPredictorFeedForward",
      "config" : dict(
        pred_hidden=128,
        ffn_intermediate=512
      ),
      "input" : (
        randTensor(1, 784, 128, dtype=dtype),
      )
    },
    # ---------------------------------------------------------------------------
    # OpenVLA SMALL sub-blocks (dino/vis=256, lm=512, n_patches=64)
    # ---------------------------------------------------------------------------
    "OpenVLADinoAttentionSmall" : {
      "class": "OpenVLADinoAttention",
      "config" : dict(
        hidden=256,
        n_heads=4,
        head_dim=64
      ),
      "input" : (
        randTensor(1, 64, 256, dtype=dtype),    # (batch, n_patches, dino_hidden)
      )
    },
    "OpenVLADinoFeedForwardSmall" : {
      "class": "OpenVLADinoFeedForward",
      "config" : dict(
        hidden=256,
        ffn_intermediate=1024
      ),
      "input" : (
        randTensor(1, 64, 256, dtype=dtype),
      )
    },
    "OpenVLAVisionAttentionSmall" : {
      "class": "OpenVLAVisionAttention",
      "config" : dict(
        hidden=256,
        n_heads=4,
        head_dim=64
      ),
      "input" : (
        randTensor(1, 64, 256, dtype=dtype),    # (batch, n_patches, vis_hidden)
      )
    },
    "OpenVLAVisionFeedForwardSmall" : {
      "class": "OpenVLAVisionFeedForward",
      "config" : dict(
        hidden=256,
        ffn_intermediate=1024
      ),
      "input" : (
        randTensor(1, 64, 256, dtype=dtype),
      )
    },
    "OpenVLAVisionProjectionSmall" : {
      "class": "OpenVLAVisionProjection",
      "config" : dict(
        proj_in=512,
        lm_hidden=512
      ),
      "input" : (
        randTensor(1, 64, 512, dtype=dtype),    # (batch, n_patches, dino+siglip concat)
      )
    },
    "OpenVLALMPrefillAttentionSmall" : {
      "class": "OpenVLALMPrefillAttention",
      "config" : dict(
        hidden=512,
        n_heads=8,
        head_dim=64
      ),
      "input" : (
        randTensor(1, 128, 512, dtype=dtype),   # (batch, vision+lang tokens, lm_hidden)
      )
    },
    "OpenVLALMFeedForwardSmall" : {
      "class": "OpenVLALMFeedForward",
      "config" : dict(
        hidden=512,
        ffn_intermediate=2048
      ),
      "input" : (
        randTensor(1, 128, 512, dtype=dtype),
      )
    },
    "OpenVLALMDecodeAttentionSmall" : {
      "class": "OpenVLALMDecodeAttention",
      "config" : dict(
        hidden=512,
        n_heads=8,
        head_dim=64,
        cache_len=128
      ),
      "input" : (
        randTensor(1, 1, 512, dtype=dtype),               # x: new action token
        randTensor(1, 8, 128, 64, dtype=dtype),            # k_cache
        randTensor(1, 8, 128, 64, dtype=dtype),            # v_cache
      )
    },
    "mobilenet" : {
      "class": "MobileNet",
      "config" : {},
      "input" : (
        randTensor(1, 3, 32, 32, dtype=dtype),
      )
    },
    "syr2k" : {
      "class": "syr2k",
      "config" : {},
      "input" : (
        randTensor(200, 200, dtype=dtype),  # C
        randTensor(200, 240, dtype=dtype),  # A
        randTensor(200, 240, dtype=dtype)   # B
      )
    },
    "vgg16" : {
      "class": "VGG16",
      "config" : {},
      "input" : (
        randTensor(1, 3, 32, 32, dtype=dtype),
      )
    },
    "sgd_sw" : {
      "class": "sgd_sw",
      "config" : dict(
        num_features=64,
        num_training=100,
        num_epochs=5,
        step_size=0.01
      ),
      "input" : (
        randTensor(6400, dtype=dtype),  # data (num_training * num_features flattened)
        randTensor(100, dtype=dtype),   # label
        randTensor(64, dtype=dtype)     # theta
      )
    },
    "backprop" : {
      "class": "backprop",
      "config" : dict(
        input_dimension=65,
        nodes_per_layer=16,
        possible_outputs=10,
        learning_rate=0.01
      ),
      "input" : (
        randTensor(32, 65, dtype=dtype),   # training_data
        randTensor(32, 10, dtype=dtype)    # training_targets
      )
    },
    "lenet" : {
      "class": "lenet",
      "config" : {},
      "input" : (
        randTensor(1, 3, 32, 32, dtype=dtype),
      )
    },
    "llama" : {
      "class": "Transformer",
      "config" : {},
      "input" : (
        torch.randint(0, 128256, (1, 8), dtype=torch.long),
      )
    },
    "Inference" : {
      "class": "Inference",
      "config" : dict(
        embed_dim=128,
        num_heads=4,
        num_layers=1,
        num_tokens=20000,
        ff_dim=256
      ),
      "input" : (
        randTensor(1, 512, 128, dtype=dtype),
      )
    },
        "gemm" : {
      "class": "gemm",
      "config" : {},
      "input" : (
        randTensor(200, 240, dtype=dtype),
        randTensor(240, 220, dtype=dtype),
        randTensor(200, 220, dtype=dtype)
      )
    },
    "k2mm" : {
      "class": "k2mm",
      "config" : {},
      "input" : (
        randTensor(180, 210, dtype=dtype),     # A
        randTensor(210, 190, dtype=dtype),     # B
        randTensor(190, 220, dtype=dtype),     # C
        randTensor(180, 220, dtype=dtype)      # D
      )
    },
    "k3mm" : {
      "class": "k3mm",
      "config" : {},
      "input" : (
        randTensor(180, 180, dtype=dtype),
        randTensor(180, 180, dtype=dtype),
        randTensor(180, 180, dtype=dtype),
        randTensor(180, 180, dtype=dtype)
      )
    },
    "k3mm_call" : {
      "class": "k3mm_call",
      "config" : {},
      "input" : (
        randTensor(180, 200, dtype=dtype),
        randTensor(200, 190, dtype=dtype),
        randTensor(190, 220, dtype=dtype),
        randTensor(220, 210, dtype=dtype)
      )
    },
    "atax" : {
      "class": "atax",
      "config" : {},
      "input" : (
        randTensor(390, 410, dtype=dtype),  # A
        randTensor(410, dtype=dtype)        # x
      )
    },
    "bicg" : {
      "class": "bicg",
      "config" : {},
      "input" : (
        randTensor(410, 390, dtype=dtype),  # A
        randTensor(410, 390, dtype=dtype),  # A copy
        randTensor(410, dtype=dtype),       # r
        randTensor(390, dtype=dtype)        # p
      )
    },
    "kernel_bicg" : {
      "class": "bicg",
      "config" : {},
      "input" : (
        randTensor(410, 390, dtype=dtype),  # A
        randTensor(410, 390, dtype=dtype),  # A copy
        randTensor(410, dtype=dtype),       # r
        randTensor(390, dtype=dtype)        # p
      )
    },
    "mvt" : {
      "class": "mvt",
      "config" : {},
      "input" : (
        randTensor(400, 400, dtype=dtype),  # A
        randTensor(400, 400, dtype=dtype),  # A copy
        randTensor(400, dtype=dtype),       # y1
        randTensor(400, dtype=dtype)        # y2
      )
    },
    "gesummv" : {
      "class": "gesummv",
      "config" : {},
      "input" : (
        randTensor(250, 250, dtype=dtype),  # A
        randTensor(250, 250, dtype=dtype),  # B
        randTensor(250, dtype=dtype)       # x
      )
    },
    "gemmv" : {
      "class": "gemmv",
      "config" : {},
      "input" : (
        randTensor(250, 250, dtype=dtype),  # A
        randTensor(250, dtype=dtype),       # x
        randTensor(250, dtype=dtype)        # y
      )
    },
    "kernel_2mm" : {
      "class": "kernel_2mm",
      "config" : {},
      "input" : (
        randTensor(200, 200, dtype=dtype),  # A
        randTensor(200, 200, dtype=dtype),  # B
        randTensor(200, 200, dtype=dtype),  # C
        randTensor(200, 200, dtype=dtype)   # D
      )
    },
    "kernel_3mm" : {
      "class": "kernel_3mm",
      "config" : {},
      "input" : (
        randTensor(180, 180, dtype=dtype),  # A
        randTensor(180, 180, dtype=dtype),  # B
        randTensor(180, 180, dtype=dtype),  # C
        randTensor(180, 180, dtype=dtype)   # D
      )
    },
    "kernel_atax" : {
      "class": "kernel_atax",
      "config" : {},
      "input" : (
        randTensor(390, 410, dtype=dtype),  # A
        randTensor(410, dtype=dtype)        # x
      )
    },
    "kernel_doitgen" : {
      "class": "kernel_doitgen",
      "config" : {},
      "input" : (
        randTensor(32, 32, 32, dtype=dtype),  # A (NR, NQ, NP)
        randTensor(32, 32, dtype=dtype)       # C4 (NP, NP)
      )
    },
    "kernel_mvt" : {
      "class": "kernel_mvt",
      "config" : {},
      "input" : (
        randTensor(400, dtype=dtype),       # x1
        randTensor(400, dtype=dtype),       # x2
        randTensor(400, dtype=dtype),       # y_1
        randTensor(400, dtype=dtype),       # y_2
        randTensor(400, 400, dtype=dtype)   # A
      )
    },
    "attention_head" : {
      "class": "attention_head",
      "config" : {},
      "input" : (
        randTensor(64, 128, dtype=dtype),  # query
        randTensor(64, 128, dtype=dtype),  # key
        randTensor(64, 128, dtype=dtype)   # value
      )
    },
    "bitnet" : {
      "class": "bitnet",
      "config" : {},
      "input" : (
        torch.randint(0, 20000, (1, 1024)),
      )
    },
    "digitrec_sw" : {
      "class": "digitrec_sw",
      "config" : dict(
        k_const=3,
        num_classes=10,
        class_size=180
      ),
      "input" : (
        torch.randint(0, 256, (1800, 196), dtype=torch.int32),  # training_set
        torch.randint(0, 256, (50, 196), dtype=torch.int32)     # test_set
      )
    }
  }
}
