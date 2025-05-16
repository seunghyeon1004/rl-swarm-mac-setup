#!/bin/bash

# RL-Swarm Mac 설치 및 최적화 스크립트
# 설명: Mac 환경에서 RL-Swarm을 설치하고 최적화 설정을 적용합니다.

# 1. 원본 RL-Swarm 저장소 클론
echo "=== RL-Swarm 저장소 클론 중... ==="
cd $HOME
if [ -d "rl-swarm" ]; then
  echo "기존 rl-swarm 디렉토리 백업 중..."
  if [ -f "rl-swarm/swarm.pem" ]; then
    echo "중요: swarm.pem 파일 백업 중..."
    cp rl-swarm/swarm.pem ~/swarm.pem.backup
  fi
  if [ -f "rl-swarm/userData.json" ]; then
    echo "중요: userData.json 파일 백업 중..."
    cp rl-swarm/userData.json ~/userData.json.backup
  fi
  if [ -f "rl-swarm/userApiKey.json" ]; then
    echo "중요: userApiKey.json 파일 백업 중..."
    cp rl-swarm/userApiKey.json ~/userApiKey.json.backup
  fi
  mv rl-swarm rl-swarm-backup-$(date +%Y%m%d%H%M%S)
fi

git clone https://github.com/gensyn-ai/rl-swarm.git
cd rl-swarm

# 백업한 파일 복원
if [ -f ~/swarm.pem.backup ]; then
  echo "swarm.pem 파일 복원 중..."
  cp ~/swarm.pem.backup swarm.pem
fi
if [ -f ~/userData.json.backup ]; then
  echo "userData.json 파일 복원 중..."
  cp ~/userData.json.backup userData.json
fi
if [ -f ~/userApiKey.json.backup ]; then
  echo "userApiKey.json 파일 복원 중..."
  cp ~/userApiKey.json.backup userApiKey.json
fi

# 2. 가상환경 설정
echo "=== 가상환경 설정 중... ==="
python3 -m venv .venv
source .venv/bin/activate

# 가상환경 활성화 검증
if [[ "$(which python)" != *".venv"* ]]; then
  echo "⚠️ 가상환경이 활성화되지 않았습니다. 수동으로 'source .venv/bin/activate' 실행 후 다시 시도하세요."
  exit 1
fi
echo "✅ 가상환경이 성공적으로 활성화되었습니다."

# 3. 필요한 패키지 설치
echo "=== 필수 패키지 설치 중... ==="
pip install --upgrade pip
pip install hivemind datasets trl peft transformers bitsandbytes accelerate
pip install torch torchvision torchaudio
pip install deepspeed psutil

# 4. 패키지 설치 확인
echo "=== 패키지 설치 확인 중... ==="
python3 -c "import hivemind" || { echo "⚠️ Hivemind 패키지 설치 실패. 스크립트를 중단합니다."; exit 1; }
echo "✅ Hivemind 패키지가 성공적으로 설치되었습니다."

# 5. 메모리 정리
echo "=== 시스템 메모리 정리 중... ==="
sudo purge

# 6. 최적화된 설정 파일 생성
echo "=== 설정 파일 생성 중... ==="
mkdir -p hivemind_exp/configs/mac
cat > hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized.yaml << 'EOFCONFIG'
# Mac 환경 최적화 설정
model_revision: main
torch_dtype: float16
bf16: false
tf32: false

# 4비트 양자화 설정
load_in_4bit: true
bnb_4bit_compute_dtype: "float16"
bnb_4bit_quant_type: "nf4"
bnb_4bit_use_double_quant: true

# 데이터셋 인수
dataset_id_or_path: 'openai/gsm8k'
dataset_config_name: "main"
max_train_samples: 30
max_eval_samples: 10

# 학습 인수
max_steps: 3
gradient_accumulation_steps: 1
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
  preserve_rng_state: false
max_grad_norm: 0.5
learning_rate: 5.0e-8
lr_scheduler_type: constant
warmup_ratio: 0.0
dataloader_num_workers: 0
dataloader_pin_memory: false
dataloader_drop_last: true
local_rank: -1
data_seed: 42
torch_compile: false

# LoRA 설정
use_peft: true
peft_config:
  r: 1
  lora_alpha: 4
  lora_dropout: 0.0
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules: ["q_proj", "v_proj"]

# GRPO 인수
use_vllm: false
num_generations: 2
per_device_train_batch_size: 1
beta: 0.0001
max_prompt_length: 96
max_completion_length: 192

# 생성 매개변수
generation_kwargs:
  temperature: 0.2
  do_sample: true
  top_k: 20
  top_p: 0.9
  repetition_penalty: 1.2
  max_new_tokens: 48

# 로깅 인수
logging_strategy: "no"
report_to:
- tensorboard
save_strategy: "no"

# 출력 디렉토리
output_dir: runs/gsm8k/multinode/Qwen-Chat-Gensyn-Swarm
EOFCONFIG

# 7. 멀티프로세싱 설정
echo "=== 멀티프로세싱 설정 중... ==="
python3 -c "
import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn')
    print('✅ MultiProcessing 시작 방법을 spawn으로 설정했습니다.')
"

# 8. Hivemind 타임아웃 값 증가
echo "=== Hivemind 타임아웃 값 설정 중... ==="
python3 -c "
import os, re
try:
    import hivemind.p2p.p2p_daemon as m
    filepath = m.__file__
    print(f'파일 수정 중: {filepath}')
    with open(filepath, 'r') as f:
        content = f.read()
    modified = re.sub(r'startup_timeout: float = [0-9]+', 'startup_timeout: float = 300', content)
    with open(filepath, 'w') as f:
        f.write(modified)
    print('✅ Hivemind 타임아웃 값을 300초로 성공적으로 수정했습니다.')
except Exception as e:
    print(f'⚠️ Hivemind 타임아웃 값 수정 실패: {e}')
    print('스크립트를 계속 진행합니다...')
"

# 9. 환경 변수 설정
echo "=== 환경 변수 설정 중... ==="
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export BITSANDBYTES_NOWELCOME=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export CONFIG_PATH="hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized.yaml"

# 10. 실행 권한 설정 및 실행
chmod +x run_rl_swarm.sh
echo "=== 모든 준비가 완료되었습니다. RL-Swarm 실행 중... ==="
./run_rl_swarm.sh
