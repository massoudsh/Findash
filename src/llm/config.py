from peft import LoraConfig, TaskType

# Model and Tokenizer Configuration
MODEL_NAME_OR_PATH = "facebook/bart-large"
TOKENIZER_NAME_OR_PATH = "facebook/bart-large"

# PEFT Configuration
PEFT_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Dataset Configuration
DATASET_NAME = "twitter_complaints"
RAFT_DATASET_SUBSET = "ought/raft"
TEXT_COLUMN = "Tweet text"
LABEL_COLUMN = "text_label"

# Training Hyperparameters
LEARNING_RATE = 3e-3
NUM_EPOCHS = 5
BATCH_SIZE = 8
SEED = 42

# Inference Configuration
MAX_BATCH_SIZE_INFERENCE = 32
MAX_WAIT_TIME_INFERENCE = 0.1  # seconds
CACHE_SIZE_INFERENCE = 1000

# Monitoring Configuration
LATENCY_WINDOW_SIZE = 1000
P95_LATENCY_ALERT_THRESHOLD = 0.1  # 100ms

# --- Unsloth & Llama Placeholder Configs ---
# This would be populated with actual model details
UNSLOTH_MODEL_NAME = "unsloth/llama-2-7b-bnb-4bit"
MAX_SEQ_LENGTH = 2048 