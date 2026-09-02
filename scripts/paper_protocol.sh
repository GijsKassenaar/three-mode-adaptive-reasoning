# Paper protocol constants. Source from training and eval jobs.
# These numbers are the 1.5B MATH run in scripts/train_three_mode_routing_math.job
# and the benchmark eval in scripts/eval_benchmarks_{baseline,all}.job.
# Comparison and scale-up jobs must keep the algorithm/eval knobs unless the
# paper protocol itself changes. GPU layout for 7B is hardware, not protocol.

PAPER_MAX_PROMPT_LENGTH=2048
PAPER_MAX_RESPONSE_LENGTH=16384

PAPER_TRAIN_BATCH_SIZE=128
PAPER_VAL_BATCH_SIZE=256
PAPER_VAL_ONLY_BATCH_SIZE=1000

PAPER_GROUP_SIZE=8
PAPER_PPO_MINI_BATCH_SIZE=32
PAPER_PPO_MICRO_BATCH_SIZE=8
PAPER_ACTOR_LR=1e-6
PAPER_LOSS_AGG_MODE=token-mean

PAPER_VAL_TEMPERATURE=0.6
PAPER_VAL_TOP_P=1.0
PAPER_VAL_DO_SAMPLE=True

PAPER_AIME_N=16
PAPER_MATH500_N=5
PAPER_GSM8K_N=5

PAPER_NOTHINK_CAP=1024
PAPER_SHORT_CAP=3000
PAPER_LONG_CAP=16384

PAPER_NOTHINK_BASE=1.3
PAPER_SHORT_BASE=1.2
PAPER_LONG_BASE=1.0
PAPER_GAMMA_NOTHINK=0.99984
PAPER_GAMMA_SHORT=0.99994
PAPER_GAMMA_LONG=1.0

PAPER_WARMUP_STEPS=45
PAPER_PHASE1_STEPS=60
PAPER_TOTAL_STEPS=90
PAPER_TEST_FREQ=10
PAPER_BALANCE_COEF=1.0
PAPER_BALANCE_TARGET=0.3333
PAPER_BALANCE_ANNEAL_END_STEP=85
PAPER_UNKNOWN_PENALTY=-0.5

PAPER_BASE_MODEL_1P5B=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
PAPER_BASE_MODEL_7B=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Released AdaptThink checkpoints (Zhang et al. 2025). Evaluated under THIS paper's
# protocol, not retrained. Their original eval also uses temperature 0.6 but top_p=0.95;
# we keep top_p=1.0 to match our other tables.
PAPER_ADAPTTHINK_1P5B=THU-KEG/AdaptThink-1.5B-delta0.05
PAPER_ADAPTTHINK_7B=THU-KEG/AdaptThink-7B-delta0.05
PAPER_AUTOTHINK_1P5B=SONGJUNTU/Distill-R1-1.5B-AutoThink-Stage3

paper_require_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "ERROR: missing $path" >&2
        echo "Run: bash scripts/prepare_math_lighteval_math500.sh && bash scripts/prepare_benchmarks.sh" >&2
        exit 1
    fi
}

# verl uses Python 3.10+ type unions (X | Y). System python3.9 fails at import.
paper_require_python() {
    echo "python3=$(command -v python3) ($(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])'))"
    python3 -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10 (conda env), got %s from %s" % (sys.version, sys.executable)'
}
