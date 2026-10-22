export interface GlossaryTermEntry {
  key: string;
  term: string;
  category: 'METRIC' | 'HYPERPARAMETER' | 'FORMAT' | 'HARDWARE' | 'ARCHITECTURE';
  whatItIs: string;
  whatItTellsUs: string;
  idealDirection?: string;
}

export const ML_GLOSSARY: Record<string, GlossaryTermEntry> = {
  ppl: {
    key: 'ppl',
    term: 'PPL (Validation Perplexity)',
    category: 'METRIC',
    whatItIs:
      'The exponentiated cross-entropy loss (exp(loss)) computed over held-out validation tokens that the model never saw during training.',
    whatItTellsUs:
      'Measures how uncertain or "surprised" the model is when predicting the next token. Lower PPL indicates higher fluency and domain accuracy. If training loss drops while PPL rises, the model is memorizing rather than generalizing.',
    idealDirection: 'Lower is better (e.g. 4.80 → 4.15 ppl)',
  },
  mmluPro: {
    key: 'mmluPro',
    term: 'MMLU-Pro (Massive Multitask Language Understanding — Pro)',
    category: 'METRIC',
    whatItIs:
      'A rigorous 10-choice evaluation benchmark spanning 14 college-level STEM, legal, medical, and reasoning disciplines.',
    whatItTellsUs:
      'Reveals whether fine-tuning preserved or enhanced broad multi-step reasoning and factual knowledge, or if narrow task training caused "catastrophic forgetting" of general intelligence.',
    idealDirection: 'Higher % is better (0–100%)',
  },
  gsm8k: {
    key: 'gsm8k',
    term: 'GSM8K (Grade School Math 8K)',
    category: 'METRIC',
    whatItIs:
      'An 8-shot chain-of-thought mathematical word problem benchmark requiring multi-step arithmetic reasoning.',
    whatItTellsUs:
      'Acts as a sensitive canary for sequential reasoning stability. Even small regressions on GSM8K signal that the learning rate or adapter rank is disrupting multi-hop logic.',
    idealDirection: 'Higher % is better (0–100%)',
  },
  humanEval: {
    key: 'humanEval',
    term: 'HumanEval+ (Pass@1 Code Synthesis)',
    category: 'METRIC',
    whatItIs:
      'An execution-based coding benchmark where generated functions are tested against strict unit test suites.',
    whatItTellsUs:
      'Tells us if the model can translate natural language specifications into syntactically and algorithmically correct code on the first attempt (Pass@1).',
    idealDirection: 'Higher % is better (0–100%)',
  },
  ifEval: {
    key: 'ifEval',
    term: 'IFEval (Instruction Following Evaluation)',
    category: 'METRIC',
    whatItIs:
      'A verifiable constraint benchmark testing strict formatting rules (e.g., "output valid JSON only", "under 150 words", "no commas").',
    whatItTellsUs:
      'Directly measures how reliably the model obeys system prompts and structural constraints in production pipelines.',
    idealDirection: 'Higher % is better (0–100%)',
  },
  mtBench: {
    key: 'mtBench',
    term: 'MT-Bench (Multi-Turn Conversational Score)',
    category: 'METRIC',
    whatItIs:
      'An LLM-as-a-Judge benchmark scoring 1–10 across two-turn follow-up dialogues covering writing, roleplay, extraction, and coding.',
    whatItTellsUs:
      'Indicates how well the model maintains context across multiple turns and follows follow-up corrections without losing coherence.',
    idealDirection: 'Higher is better (1.0 – 10.0 scale)',
  },
  trainValLoss: {
    key: 'trainValLoss',
    term: 'Cross-Entropy Training & Validation Loss',
    category: 'METRIC',
    whatItIs:
      'The negative log-likelihood penalty measuring the divergence between the model’s predicted token probabilities and actual target tokens.',
    whatItTellsUs:
      'Training loss shows how fast the model fits the current dataset; Validation loss shows how well it generalizes to unseen data. A widening gap (Val >> Train) warns of overfitting.',
    idealDirection: 'Smooth downward convergence (< 1.0)',
  },
  gradNorm: {
    key: 'gradNorm',
    term: 'Gradient L2 Norm (||g||₂)',
    category: 'METRIC',
    whatItIs:
      'The Euclidean magnitude of weight update gradients across all trainable adapter matrices during backpropagation.',
    whatItTellsUs:
      'Tells us if training is numerically stable. A sudden spike (> 2.5) indicates exploding gradients from bad data samples or an excessive learning rate; near-zero means vanishing updates.',
    idealDirection: 'Stable between 0.35 and 1.25',
  },
  loraRank: {
    key: 'loraRank',
    term: 'LoRA Rank (r) & Alpha (α)',
    category: 'HYPERPARAMETER',
    whatItIs:
      'Rank (r) defines the inner dimension of low-rank matrices A and B injected into transformer layers; Alpha (α) scales their output via (α / r) · (B @ A).',
    whatItTellsUs:
      'Controls adapter capacity vs. memory footprint. For sub-2B models, r=16 to r=32 captures new domain skills without overfitting; α=2r keeps gradient magnitudes balanced.',
    idealDirection: 'Sub-2B sweet spot: r=16–32, α=32–64',
  },
  learningRate: {
    key: 'learningRate',
    term: 'Peak Learning Rate (LR)',
    category: 'HYPERPARAMETER',
    whatItIs:
      'The step size multiplier applied by the AdamW optimizer when updating trainable adapter weights.',
    whatItTellsUs:
      'Too high (> 5e-4) causes loss spikes and destroys base model knowledge; too low (< 2e-5) results in slow underfitting. Compact sub-2B models typically thrive around 1.5e-4 to 2.5e-4.',
    idealDirection: '1.5e-4 – 2.5e-4 for QLoRA',
  },
  wsdScheduler: {
    key: 'wsdScheduler',
    term: 'LR Scheduler (WSD / Cosine)',
    category: 'HYPERPARAMETER',
    whatItIs:
      'The curve governing how the learning rate warms up, plateaus, and decays across training steps (e.g., Warmup-Stable-Decay or Cosine Annealing).',
    whatItTellsUs:
      'WSD maintains high plasticity during the main phase and rapidly anneals in the final 15% of steps, consistently outperforming standard cosine decay on compact ≤2B models.',
    idealDirection: 'WSD or Cosine with 5–8% warmup',
  },
  neftune: {
    key: 'neftune',
    term: 'NEFTune Noise Alpha (α)',
    category: 'HYPERPARAMETER',
    whatItIs:
      'Noisy Embeddings Fine-Tuning: injects uniform random noise scaled by α/√(L·d) into token embeddings during forward passes.',
    whatItTellsUs:
      'Prevents small models from memorizing exact phrasing in instruction datasets, boosting conversational quality (MT-Bench) and instruction adherence by 3–6%.',
    idealDirection: 'α = 5 for instruction tuning',
  },
  ggufQ4KM: {
    key: 'ggufQ4KM',
    term: 'GGUF-Q4_K_M (4-Bit K-Quant Medium)',
    category: 'FORMAT',
    whatItIs:
      'A single-file binary format for llama.cpp and Ollama using 4-bit super-block quantization with 6-bit precision preserved on critical attention/feed-forward tensors.',
    whatItTellsUs:
      'Shrinks a 2B model to ~1.25 GB so it runs at 90–140 tok/s on consumer CPUs, Apple Silicon, or edge GPUs while retaining >99% of BF16 accuracy.',
  },
  gptq: {
    key: 'gptq',
    term: 'GPTQ (Post-Training GPU Quantization)',
    category: 'FORMAT',
    whatItIs:
      'A second-order Hessian-calibrated 4-bit weight quantization format optimized for high-throughput CUDA/TensorRT-LLM and vLLM GPU kernels (Marlin/ExLlamaV2).',
    whatItTellsUs:
      'Ideal when deploying fine-tuned models onto dedicated NVIDIA GPUs where maximum concurrent batch throughput and minimal VRAM footprint are required.',
  },
  onnx: {
    key: 'onnx',
    term: 'ONNX (Open Neural Network Exchange)',
    category: 'FORMAT',
    whatItIs:
      'A cross-platform static computation graph format with fused operator nodes for ONNX Runtime, DirectML, CoreML, WebGPU, and mobile NPUs.',
    whatItTellsUs:
      'Allows deploying your fine-tuned sub-2B model directly inside browser WebGPU apps, Windows DirectML, Android/iOS NPUs, or enterprise C++/C# microservices without Python.',
  },
  safetensors: {
    key: 'safetensors',
    term: 'Safetensors (Zero-Copy Contiguous Format)',
    category: 'FORMAT',
    whatItIs:
      'HuggingFace’s secure, memory-mapped (mmap) tensor serialization format storing an 8-byte aligned JSON header followed by raw contiguous weight buffers.',
    whatItTellsUs:
      'Eliminates arbitrary Python pickle code execution vulnerabilities and loads weights into GPU/RAM up to 10x faster via zero-copy OS page mapping.',
  },
  smOccupancy: {
    key: 'smOccupancy',
    term: 'SM Compute Occupancy & Tensor Core Pipe',
    category: 'HARDWARE',
    whatItIs:
      'The percentage of GPU Streaming Multiprocessors (SMs) and matrix-multiply Tensor Cores actively executing warps per clock cycle.',
    whatItTellsUs:
      'High SM/Tensor Core utilization (>80%) means your pipeline is compute-bound (optimal). Low occupancy (<50%) while VRAM is full means the CPU dataloader or PCIe bus is bottlenecking training.',
    idealDirection: '80% – 95% sustained occupancy',
  },
  chatml: {
    key: 'chatml',
    term: 'ChatML / ShareGPT / Alpaca Schema',
    category: 'FORMAT',
    whatItIs:
      'Standardized conversation framing templates (e.g. <|im_start|>role ... <|im_end|> in ChatML) that separate system instructions, user queries, and assistant responses.',
    whatItTellsUs:
      'Normalizing disparate HuggingFace datasets into a single canonical chat template prevents prompt-boundary confusion and stop-token leaks during inference.',
  },
  lfmArchitecture: {
    key: 'lfmArchitecture',
    term: 'LFM2.5 / Qwen3.5 / Gemma4 Architectures',
    category: 'ARCHITECTURE',
    whatItIs:
      'LFM2.5 uses Liquid Hybrid State-Space + Conv1D blocks; Qwen3.5 uses GQA + SwiGLU with Unsloth RoPE; Gemma4 uses interleaved local/global attention with logit soft-capping.',
    whatItTellsUs:
      'Determines KV-cache scaling and target LoRA projection modules. LFM2.5 uses ~28% less activation VRAM on long contexts than standard dense transformers.',
  },
  qlora4bit: {
    key: 'qlora4bit',
    term: 'QLoRA (4-Bit NormalFloat NF4 + Double Quantization)',
    category: 'HYPERPARAMETER',
    whatItIs:
      'Freezes the base model in information-theoretically optimal 4-bit NormalFloat (NF4) precision while backpropagating gradients only through BF16 low-rank LoRA adapters.',
    whatItTellsUs:
      'Cuts base weight VRAM by 75% so sub-2B and 4B models can be fine-tuned with long contexts (8k–32k) on a single consumer GPU with virtually zero accuracy loss vs. 16-bit LoRA.',
    idealDirection: 'Default for local sub-2B & 4B fine-tuning',
  },
  gradCheckpointing: {
    key: 'gradCheckpointing',
    term: 'Gradient (Activation) Checkpointing',
    category: 'HARDWARE',
    whatItIs:
      'Discards intermediate forward-pass layer activations from VRAM and recomputes them on-the-fly during the backward pass.',
    whatItTellsUs:
      'Reduces activation memory footprint by 60–75% at the cost of ~15% extra compute time, preventing CUDA Out-Of-Memory (OOM) errors on long-context sequences.',
  },
  minhashDedup: {
    key: 'minhashDedup',
    term: 'MinHash LSH Deduplication & PII Scrubbing',
    category: 'METRIC',
    whatItIs:
      'Locality-Sensitive Hashing (Jaccard n-gram similarity) combined with regex/NER pattern scanners across ingested training corpora.',
    whatItTellsUs:
      'Eliminates near-duplicate prompts that cause small models to overfit and repeat phrases verbatim, while preventing accidental memorization of emails, keys, or SSNs.',
    idealDirection: '0 PII leaks · 0 near-duplicate collisions',
  },
  orpoAlignment: {
    key: 'orpoAlignment',
    term: 'Stage 2 Preference Alignment (ORPO / SimPO / DPO / KTO)',
    category: 'HYPERPARAMETER',
    whatItIs:
      'Post-SFT preference optimization using Chosen vs. Rejected response pairs. ORPO (Odds Ratio) and SimPO optimize preference margins directly without loading a frozen reference model.',
    whatItTellsUs:
      'Dramatically improves IFEval constraint compliance, MT-Bench multi-turn coherence, and reasoning faithfulness in sub-2B models while saving ~45% VRAM with reference-free ORPO/SimPO.',
    idealDirection: 'ORPO (beta=0.1) or SimPO (gamma=0.5) for sub-2B models',
  },
  teacherDistillation: {
    key: 'teacherDistillation',
    term: 'On-Policy Teacher Distillation & LLM-as-a-Judge Rubric',
    category: 'METRIC',
    whatItIs:
      'Generates synthetic <think>...</think> reasoning chains and verified answers from a frontier teacher LLM, filtered by a multi-axis Judge Rubric (Faithfulness, Formatting, Conciseness).',
    whatItTellsUs:
      'Sub-2B models (0.3B–2.6B) learn 3–4x faster from clean, dense teacher traces than noisy web scrapes; rubric filtering drops hallucinated or verbose synthetic rows before training.',
    idealDirection: 'Judge Rubric Score >= 8.5 / 10',
  },
  adapterMerging: {
    key: 'adapterMerging',
    term: 'Multi-Adapter Task Arithmetic (DARE-TIES / TIES / SLERP)',
    category: 'ARCHITECTURE',
    whatItIs:
      'Combines multiple task-specialized LoRA adapters (e.g., Code + Math + Tool-Calling) via Drop-And-REscale sign-consensus (DARE-TIES) or Spherical Linear Interpolation (SLERP).',
    whatItTellsUs:
      'Prevents multi-task gradient interference in small models by training focused adapters separately and merging their delta weights with sign-conflict resolution.',
    idealDirection: 'DARE-TIES (density 0.75–0.85) for 3+ adapters; SLERP for 2 adapters',
  },
  ifdPruning: {
    key: 'ifdPruning',
    term: 'IFD Perplexity Pruning & 13-Gram Benchmark Decontamination',
    category: 'METRIC',
    whatItIs:
      'Instruction Following Difficulty (IFD) scores conditioned vs. unconditioned perplexity per sample, paired with 13-gram overlap scanning against MMLU-Pro, GSM8K, and HumanEval+.',
    whatItTellsUs:
      'Applies the LIMA Less-Is-More principle: dropping corrupted high-PPL outliers (IFD > 1.05) and trivial rows (IFD < 0.15) boosts sub-2B generalization while preventing test set leakage.',
    idealDirection: 'Retain IFD 0.20–0.92 · 0% benchmark overlap',
  },
  sequencePacking: {
    key: 'sequencePacking',
    term: 'FlashAttention Multipack & Completion-Only Loss Masking',
    category: 'HYPERPARAMETER',
    whatItIs:
      'Concatenates multiple variable-length examples into a single context window with block-diagonal attention masks, while setting loss labels to -100 on all user prompt tokens.',
    whatItTellsUs:
      'Eliminates up to 68% <pad> token compute waste and ensures 100% of gradient updates optimize assistant responses rather than memorizing user prompts.',
    idealDirection: 'Multipack ON + Completion-Only Masking ON (+2.5–3.8% sub-2B accuracy)',
  },
  earlyStopping: {
    key: 'earlyStopping',
    term: 'Validation Loss Patience Watchdog & Best Checkpoint Rollback',
    category: 'METRIC',
    whatItIs:
      'Monitors held-out validation loss after every evaluation step, automatically snapshotting the lowest-val-loss checkpoint and halting if val loss fails to improve for N patience steps.',
    whatItTellsUs:
      'Prevents late-epoch overfitting (common in 0.8B–2B models around Epoch 2.0+) and guarantees your exported Safetensors/GGUF checkpoint comes from the peak generalization step.',
    idealDirection: 'Patience = 2–3 eval steps · Min Delta = 0.005',
  },
  imatrixCalibration: {
    key: 'imatrixCalibration',
    term: 'Activation-Aware Importance Matrix (imatrix) Calibration',
    category: 'FORMAT',
    whatItIs:
      'Computes per-tensor activation magnitudes across 128–512 domain samples before quantizing FP16 weights to GGUF Q4_K_M / IQ4_XS, protecting salient outlier channels from rounding error.',
    whatItTellsUs:
      'Sub-2B models (0.3B–2B) suffer steep perplexity spikes under naive 4-bit quantization. Calibrating an imatrix on your active Dataset Studio blend recovers 70%–85% of lost quantization accuracy.',
    idealDirection: 'Recover >= 75% Q4_K_M PPL loss · Protect embed/lm_head/SSM projections',
  },
  grpoRlvr: {
    key: 'grpoRlvr',
    term: 'Stage 3 GRPO (Group Relative Policy Optimization) & Verifiable Rewards',
    category: 'HYPERPARAMETER',
    whatItIs:
      'Critic-free reinforcement learning (used in DeepSeek-R1) that samples G completions per prompt and computes advantage relative to the group mean using deterministic code/math/JSON verifiers.',
    whatItTellsUs:
      'Requires 0 GB of extra Critic-model VRAM while teaching sub-2B models to self-verify JSON schemas, pass pytest unit tests, and solve multi-step math problems reliably.',
    idealDirection: 'Rollout G = 4–8 · KL Beta = 0.04 · Verifier Pass Rate >= 90%',
  },
  curriculumLearning: {
    key: 'curriculumLearning',
    term: '3-Phase Curriculum Pacing & Annealing Scheduler',
    category: 'HYPERPARAMETER',
    whatItIs:
      'Orders training samples from low-IFD short syntax alignment (Phase 1 Warmup) to multi-turn domain blends (Phase 2 Core) to high-IFD <think> traces during WSD learning-rate decay (Phase 3 Annealing).',
    whatItTellsUs:
      'Prevents early gradient spikes in sub-2B models by establishing formatting stability before introducing dense multi-step reasoning chains.',
    idealDirection: 'Warmup (0–30%) -> Core (30–80%) -> CoT Annealing (80–100%)',
  },
  doraRankAllocation: {
    key: 'doraRankAllocation',
    term: 'DoRA (Weight-Decomposed LoRA) & Layer-Wise SVD Rank Allocation',
    category: 'ARCHITECTURE',
    whatItIs:
      'Decomposes pretrained weights into magnitude and direction vectors (DoRA) while allocating higher LoRA ranks (r=32–64) to upper reasoning/MLP layers and lower ranks (r=8–16) to shallow layers.',
    whatItTellsUs:
      'Concentrates trainable parameters where abstract reasoning happens, improving MMLU-Pro and HumanEval+ by +1.8% to +3.2% at the exact same adapter file size.',
    idealDirection: 'DoRA + rsLoRA ON · Top-Heavy SVD Rank Profile',
  },
  tokenizerSurgery: {
    key: 'tokenizerSurgery',
    term: 'Tokenizer Special-Token Surgery & EOS Alignment Verifier',
    category: 'ARCHITECTURE',
    whatItIs:
      'Inspects whether control tokens (<|im_start|>, <think>, <tool_call>) tokenize into single atomic IDs versus fragmented subwords, and verifies EOS/PAD alignment across Qwen3.5, LFM2.5, and Gemma4.',
    whatItTellsUs:
      'Prevents the #1 deployment bug in small fine-tuned models: infinite generation loops caused by unmasked or fragmented EOS / <think> tokens.',
    idealDirection: '100% Atomic Special Tokens · Aligned EOS Token ID',
  },
  redTeamSuite: {
    key: 'redTeamSuite',
    term: 'Automated Red-Teaming & Edge-Case Stress-Test Matrix',
    category: 'BENCHMARK',
    whatItIs:
      'Executes standardized adversarial probes (prompt injection, malformed JSON inputs, empty context traps, and high-temperature repetition stress tests) against Base vs. Fine-Tuned checkpoints.',
    whatItTellsUs:
      'Exposes brittle failure modes that aggregate benchmarks miss, and lets you convert any failed edge case into a corrective training sample in one click.',
    idealDirection: 'Resilience Score >= 90% · 0 Repetition Loops',
  },
};


