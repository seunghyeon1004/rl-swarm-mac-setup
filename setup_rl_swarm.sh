#!/bin/bash

# RL-Swarm Mac 설치 및 최적화 스크립트
# GitHub에서 바로 실행 가능: curl -s https://raw.githubusercontent.com/seunghyeon1004/rl-swarm-mac-setup/main/setup_rl_swarm.sh | bash

# 오류 발생 시 스크립트 중단
set -e

# 컬러 출력 설정
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 성공 메시지 함수
success() {
  echo -e "${GREEN}✅ $1${NC}"
}

# 경고 메시지 함수
warn() {
  echo -e "${YELLOW}⚠️ $1${NC}"
}

# 오류 메시지 함수
error() {
  echo -e "${RED}❌ $1${NC}"
  exit 1
}

echo "
    ██████  ██            ███████ ██     ██  █████  ██████  ███    ███
    ██   ██ ██            ██      ██     ██ ██   ██ ██   ██ ████  ████
    ██████  ██      █████ ███████ ██  █  ██ ███████ ██████  ██ ████ ██
    ██   ██ ██                 ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██
    ██   ██ ███████       ███████  ███ ███  ██   ██ ██   ██ ██      ██

    Mac용 RL-Swarm 설치 도구
"

# 1. 원본 RL-Swarm 저장소 클론
echo "=== RL-Swarm 저장소 클론 중... ==="
cd $HOME
if [ -d "rl-swarm" ]; then
  echo "기존 rl-swarm 디렉토리 백업 중..."
  
  # 중요 파일 백업
  for file in swarm.pem userData.json userApiKey.json; do
    if [ -f "rl-swarm/$file" ]; then
      echo "중요: $file 파일 백업 중..."
      cp "rl-swarm/$file" ~/"$file.backup"
    fi
  done
  
  # 기존 디렉토리 이름 변경
  mv rl-swarm "rl-swarm-backup-$(date +%Y%m%d%H%M%S)"
fi

# 저장소 클론
git clone https://github.com/gensyn-ai/rl-swarm.git || error "RL-Swarm 저장소 클론 실패"
cd rl-swarm || error "rl-swarm 디렉토리로 이동 실패"

# 백업 파일 복원
for file in swarm.pem userData.json userApiKey.json; do
  if [ -f ~/"$file.backup" ]; then
    echo "$file 파일 복원 중..."
    cp ~/"$file.backup" "$file"
  fi
done

# 2. 가상환경 설정
echo "=== 가상환경 설정 중... ==="
python3 -m venv .venv || error "가상환경 생성 실패"
source .venv/bin/activate || error "가상환경 활성화 실패"

# 가상환경 활성화 검증
if [[ "$(which python)" != *".venv"* ]]; then
  error "가상환경이 활성화되지 않았습니다. 수동으로 'source .venv/bin/activate' 실행 후 다시 시도하세요."
fi
success "가상환경이 성공적으로 활성화되었습니다."

# 3. 필요한 패키지 설치
echo "=== 필수 패키지 설치 중... ==="
pip install --upgrade pip || warn "pip 업그레이드 실패, 계속 진행합니다..."
pip install hivemind datasets trl peft transformers bitsandbytes accelerate || error "기본 패키지 설치 실패"
pip install torch torchvision torchaudio || warn "PyTorch 패키지 설치 실패, 계속 진행합니다..."
pip install deepspeed psutil || warn "추가 패키지 설치 실패, 계속 진행합니다..."

# 4. 패키지 설치 확인
echo "=== 패키지 설치 확인 중... ==="
python3 -c "import hivemind" || error "Hivemind 패키지 설치 실패. 스크립트를 중단합니다."
success "Hivemind 패키지가 성공적으로 설치되었습니다."

# 5. 메모리 정리
echo "=== 시스템 메모리 정리 중... ==="
sudo purge || warn "메모리 정리 실패, 계속 진행합니다..."

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
" || warn "멀티프로세싱 설정 실패, 계속 진행합니다..."

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
" || warn "Hivemind 타임아웃 설정 실패, 계속 진행합니다..."

# 9. 래퍼 스크립트 생성 (run_rl_swarm.sh 실행 시 오류 무시)
echo "=== RL-Swarm 실행 래퍼 스크립트 생성 중... ==="
cat > run_wrapper.sh << 'EOFWRAPPER'
#!/bin/bash
# RL-Swarm 실행 래퍼 스크립트 - 오류 메시지 처리
cd "$HOME/rl-swarm"
source .venv/bin/activate

# 환경 변수 설정
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export BITSANDBYTES_NOWELCOME=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export CONFIG_PATH="hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized.yaml"
export PYTORCH_METAL_GPU_SESSION="1" 
export PYTORCH_MPS_MAX_ALLOC_SIZE=4294967296

echo "RL-Swarm 실행 중... (오류 메시지는 무시됩니다)"
# 2>&1 출력을 파이프로 전달하여 오류 메시지 필터링
./run_rl_swarm.sh 2>&1 | grep -v "kill: (-[0-9]*) - No such process"

# 항상 성공 코드 반환
exit 0
EOFWRAPPER

chmod +x run_wrapper.sh
success "RL-Swarm 래퍼 스크립트 생성 완료"

# 10. 환경 변수 설정
echo "=== 환경 변수 설정 중... ==="
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export BITSANDBYTES_NOWELCOME=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export CONFIG_PATH="hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized.yaml"
export PYTORCH_METAL_GPU_SESSION="1" 
export PYTORCH_MPS_MAX_ALLOC_SIZE=4294967296

# 11. 실행 권한 설정
chmod +x run_rl_swarm.sh || warn "실행 권한 설정 실패, 계속 진행합니다..."

# 12. 사용법 안내
echo "
=== RL-Swarm 설치가 완료되었습니다 ===

RL-Swarm 실행 방법:

1. 래퍼 스크립트 사용 (오류 메시지 없이 실행):
   cd ~/rl-swarm
   ./run_wrapper.sh

2. 직접 실행 (오류 메시지 발생 가능):
   cd ~/rl-swarm
   source .venv/bin/activate
   ./run_rl_swarm.sh

RL-Swarm을 지금 실행하시겠습니까? (y/n): "

read -r answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
  echo "=== RL-Swarm 실행 중... ==="
  # 서브셸에서 실행하여 환경 변수와 가상환경 유지
  (cd "$HOME/rl-swarm" && ./run_wrapper.sh)
  success "RL-Swarm 실행이 완료되었습니다."
else
  echo "RL-Swarm을 나중에 실행하려면 다음 명령어를 사용하세요:"
  echo "cd ~/rl-swarm && ./run_wrapper.sh"
fi

echo "감사합니다! RL-Swarm 설치 도구를 이용해 주셔서 감사합니다."
