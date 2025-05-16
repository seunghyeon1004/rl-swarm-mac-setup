#!/bin/bash
# RL-Swarm Mac 완전 자동화 설치 및 최적화 스크립트 v1.0
# 클론, 설치, 최적화를 한 번에 진행하는 원스톱 솔루션

# 색상 설정
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 유틸리티 함수
print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# 배너 출력
cat << "EOF"
    ██████  ██            ███████ ██     ██  █████  ██████  ███    ███
    ██   ██ ██            ██      ██     ██ ██   ██ ██   ██ ████  ████
    ██████  ██      █████ ███████ ██  █  ██ ███████ ██████  ██ ████ ██
    ██   ██ ██                 ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██
    ██   ██ ███████       ███████  ███ ███  ██   ██ ██   ██ ██      ██

    Mac 원스톱 설치 및 최적화 도구 v1.0
EOF
echo ""

print_header "시스템 요구사항 확인"
# Python 3.10 이상 확인
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
if (( $(echo "$python_version < 3.10" | bc -l) )); then
    print_error "Python 3.10 이상이 필요합니다. 현재 버전: $python_version"
    print_warning "Python을 업그레이드한 후 다시 시도하세요."
    exit 1
else
    print_success "Python 버전 확인 완료: $python_version"
fi

# 필수 패키지 확인
print_header "필수 패키지 확인"
for cmd in git curl python3 pip3; do
    if ! command -v $cmd &> /dev/null; then
        print_error "$cmd 명령을 찾을 수 없습니다. 설치 후 다시 시도하세요."
        exit 1
    fi
    print_success "$cmd 확인 완료"
done

# 하드웨어 확인
print_header "Mac 하드웨어 확인"
mac_model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || system_profiler SPHardwareDataType | grep "Model Name" | awk -F': ' '{print $2}')
mac_chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || system_profiler SPHardwareDataType | grep "Chip" | awk -F': ' '{print $2}')
total_memory=$(sysctl -n hw.memsize 2>/dev/null || system_profiler SPHardwareDataType | grep "Memory" | awk -F': ' '{print $2}')
total_memory_gb=$(($(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024))

echo "모델: $mac_model"
echo "칩: $mac_chip"
echo "메모리: ${total_memory_gb}GB"

# M 시리즈 확인
if [[ "$mac_chip" == *"Apple"* && "$mac_chip" == *"M"* ]]; then
    print_success "Apple Silicon M 시리즈 칩 감지됨: $mac_chip"
    is_m_series=true
    
    # M1/M2/M3/M4 구분
    if [[ "$mac_chip" == *"M1"* ]]; then
        mac_series="M1"
    elif [[ "$mac_chip" == *"M2"* ]]; then
        mac_series="M2"
    elif [[ "$mac_chip" == *"M3"* ]]; then
        mac_series="M3"
    elif [[ "$mac_chip" == *"M4"* ]]; then
        mac_series="M4"
    else
        mac_series="M-unknown"
    fi
    
    print_success "감지된 칩 시리즈: $mac_series"
else
    print_warning "Apple Silicon M 시리즈 칩이 감지되지 않았습니다. 일부 최적화가 작동하지 않을 수 있습니다."
    is_m_series=false
    mac_series="non-M"
fi

# RL-Swarm 리포지토리 클론
print_header "RL-Swarm 리포지토리 클론"
cd $HOME
if [ -d "rl-swarm" ]; then
    print_warning "기존 rl-swarm 디렉토리가 존재합니다."
    read -p "기존 디렉토리를 백업하고 새로 클론하시겠습니까? (y/N): " clone_choice
    if [[ "$clone_choice" =~ ^[Yy]$ ]]; then
        echo "기존 rl-swarm 디렉토리 백업 중..."
        # 중요 파일 백업
        for file in swarm.pem userData.json userApiKey.json; do
            if [ -f "rl-swarm/$file" ]; then
                echo "중요: $file 파일 백업 중..."
                cp "rl-swarm/$file" ~/"$file.backup"
            fi
        done
        mv rl-swarm "rl-swarm-backup-$(date +%Y%m%d%H%M%S)"
        git clone https://github.com/gensyn-ai/rl-swarm.git
        print_success "RL-Swarm 리포지토리 클론 완료"
        
        # 백업 파일 복원
        for file in swarm.pem userData.json userApiKey.json; do
            if [ -f ~/"$file.backup" ]; then
                echo "$file 파일 복원 중..."
                cp ~/"$file.backup" "rl-swarm/$file"
            fi
        done
    else
        print_success "기존 rl-swarm 디렉토리를 사용합니다."
    fi
else
    git clone https://github.com/gensyn-ai/rl-swarm.git
    print_success "RL-Swarm 리포지토리 클론 완료"
fi

# RL-Swarm 디렉토리로 이동
cd ~/rl-swarm || { print_error "rl-swarm 디렉토리를 찾을 수 없습니다."; exit 1; }

# 가상환경 설정
print_header "가상환경 설정"
if [ -d ".venv" ]; then
    print_warning "기존 가상환경이 존재합니다."
    read -p "가상환경을 재생성하시겠습니까? (y/N): " venv_choice
    if [[ "$venv_choice" =~ ^[Yy]$ ]]; then
        rm -rf .venv
        python3 -m venv .venv
        print_success "가상환경이 재생성되었습니다."
    else
        print_success "기존 가상환경을 사용합니다."
    fi
else
    python3 -m venv .venv
    print_success "가상환경이 생성되었습니다."
fi

# 가상환경 활성화
source .venv/bin/activate || { print_error "가상환경 활성화 실패"; exit 1; }
print_success "가상환경 활성화 성공: $(which python)"

# 필요한 패키지 설치
print_header "필수 패키지 설치"
pip install --upgrade pip

# Apple Silicon용 패키지 설치
if [ "$is_m_series" = true ]; then
    print_header "Apple Silicon용 패키지 설치"
    pip install hivemind datasets trl peft transformers bitsandbytes accelerate
    pip install torch torchvision torchaudio
    pip install deepspeed psutil
else
    # 일반 패키지 설치
    pip install -r requirements-cpu.txt
fi

# 패키지 설치 확인
print_header "패키지 설치 확인"
python3 -c "import hivemind" || { print_error "Hivemind 패키지 설치 실패. 스크립트를 중단합니다."; exit 1; }
print_success "Hivemind 패키지가 성공적으로 설치되었습니다."

# 메모리 정리
print_header "시스템 메모리 정리"
sudo purge

# Mac 최적화 설정 디렉토리 생성
print_header "Mac 최적화 설정 생성"
mkdir -p hivemind_exp/configs/mac
cat > hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized-$mac_series.yaml << EOFCONFIG
# $mac_series Mac 환경 최적화 설정
model_revision: main
torch_dtype: float16
bf16: false
tf32: false

# 4비트 양자화 설정 (최적화됨)
load_in_4bit: true
bnb_4bit_compute_dtype: "float16"
bnb_4bit_quant_type: "nf4"
bnb_4bit_use_double_quant: true

# 데이터셋 인수
dataset_id_or_path: 'openai/gsm8k'
dataset_config_name: "main"
max_train_samples: 30
max_eval_samples: 10

# 학습 인수 (Mac 최적화)
max_steps: 3
gradient_accumulation_steps: 1
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
  preserve_rng_state: false
max_grad_norm: 0.5  # Mac 안정성 증가
learning_rate: 5.0e-8
lr_scheduler_type: constant
warmup_ratio: 0.0
dataloader_num_workers: 0  # Mac에서는 0이 좋음
dataloader_pin_memory: false
dataloader_drop_last: true
local_rank: -1
data_seed: 42
torch_compile: false

# LoRA 최적화 설정
use_peft: true
peft_config:
  r: 1
  lora_alpha: 4
  lora_dropout: 0.0
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]  # 확장된 타겟 모듈

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
output_dir: runs/gsm8k/multinode/Qwen-Chat-Gensyn-Swarm-$mac_series
EOFCONFIG

print_success "Mac $mac_series 최적화 설정 파일 생성됨: hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized-$mac_series.yaml"

# Hivemind 타임아웃 값 수정
print_header "Hivemind 타임아웃 값 설정"
python3 -c "
import os, re
try:
    import hivemind.p2p.p2p_daemon as m
    filepath = m.__file__
    print(f'파일 확인: {filepath}')
    with open(filepath, 'r') as f:
        content = f.read()
    if 'startup_timeout: float = 300' in content:
        print('✓ 이미 타임아웃 값이 300초로 설정되어 있습니다.')
    else:
        modified = re.sub(r'startup_timeout: float = [0-9]+', 'startup_timeout: float = 300', content)
        with open(filepath, 'w') as f:
            f.write(modified)
        print('✓ Hivemind 타임아웃 값을 300초로 수정했습니다.')
except Exception as e:
    print(f'⚠ Hivemind 타임아웃 값 수정 실패: {e}')
" || print_warning "Hivemind 타임아웃 값 설정 실패"

# MPS 백엔드 최적화
print_header "MPS 백엔드 최적화"
if [ "$is_m_series" = true ]; then
    # 메모리 청소 스크립트 생성
    cat > clean_memory.sh << 'EOF'
#!/bin/bash
# 메모리 정리 스크립트
echo "메모리 정리 중..."
sudo purge
vm_stat | grep "Pages free"
EOF
    chmod +x clean_memory.sh
    print_success "메모리 청소 스크립트 생성: clean_memory.sh"
    
    # M 시리즈 최적화 스크립트 생성
    cat > optimize_mps.sh << EOF
#!/bin/bash
# $mac_series MPS 백엔드 최적화 스크립트
echo "MPS 백엔드 최적화 중..."

# 메모리 관리 최적화
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_METAL_GPU_SESSION="1"

# 메모리 크기 최적화
if [ $total_memory_gb -ge 32 ]; then
    # 32GB 이상 메모리
    export PYTORCH_MPS_MAX_ALLOC_SIZE=8589934592  # 8GB
elif [ $total_memory_gb -ge 16 ]; then
    # 16GB 메모리
    export PYTORCH_MPS_MAX_ALLOC_SIZE=4294967296  # 4GB
else
    # 16GB 미만 메모리
    export PYTORCH_MPS_MAX_ALLOC_SIZE=2147483648  # 2GB
fi

# MPS 성능 최적화
export PYTORCH_MPS_PREFETCH_BATCHES=2
export PYTORCH_MPS_DEVICE_BATCH_SIZE="auto"
export PYTORCH_MPS_EXECUTOR_THREADS=4

echo "MPS 백엔드 최적화 완료"
EOF
    chmod +x optimize_mps.sh
    print_success "MPS 백엔드 최적화 스크립트 생성: optimize_mps.sh"
else
    print_warning "Apple Silicon M 시리즈가 아니므로 MPS 백엔드 최적화를 건너뜁니다."
fi

# 환경 변수 설정 스크립트 생성
print_header "환경 변수 설정 스크립트 생성"
cat > set_environment.sh << 'EOF'
#!/bin/bash
# RL-Swarm 환경 변수 설정

# 기본 환경 변수
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export BITSANDBYTES_NOWELCOME=1
export HF_HUB_DOWNLOAD_TIMEOUT=600

# Mac 최적화 (설정 파일 경로 포함)
if [ -f "./hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized-M4.yaml" ]; then
    export CONFIG_PATH="./hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized-M4.yaml"
elif [ -f "./hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized-M3.yaml" ]; then
    export CONFIG_PATH="./hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized-M3.yaml"
elif [ -f "./hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized-M2.yaml" ]; then
    export CONFIG_PATH="./hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized-M2.yaml"
elif [ -f "./hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized-M1.yaml" ]; then
    export CONFIG_PATH="./hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized-M1.yaml"
else
    export CONFIG_PATH="./hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-optimized-non-M.yaml"
fi

# MPS 백엔드 최적화
if [ -f "./optimize_mps.sh" ]; then
    source ./optimize_mps.sh
fi

echo "환경 변수가 설정되었습니다. 설정 파일: $CONFIG_PATH"
EOF
chmod +x set_environment.sh
print_success "환경 변수 설정 스크립트 생성: set_environment.sh"

# 실행 스크립트 생성
print_header "실행 스크립트 생성"
cat > run_optimized.sh << 'EOF'
#!/bin/bash
# RL-Swarm 최적화 실행 스크립트

# 색상 설정
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==== RL-Swarm 최적화 실행 ====${NC}"

# 1. 메모리 정리
if [ -f "./clean_memory.sh" ]; then
    echo -e "${YELLOW}메모리 정리 중...${NC}"
    ./clean_memory.sh
fi

# 2. 환경 변수 설정
echo -e "${YELLOW}환경 변수 설정 중...${NC}"
source ./set_environment.sh

# 3. 가상환경 활성화
echo -e "${YELLOW}가상환경 활성화 중...${NC}"
source .venv/bin/activate

# 4. 실행
echo -e "${GREEN}RL-Swarm 실행 중...${NC}"
./run_rl_swarm.sh || {
    echo -e "${YELLOW}오류가 발생했습니다. 프로세스 ID 오류는 무시해도 됩니다.${NC}"
}

echo -e "${GREEN}실행 완료${NC}"
EOF
chmod +x run_optimized.sh
print_success "최적화 실행 스크립트 생성: run_optimized.sh"

# 백업 스크립트 생성
print_header "백업 유틸리티 생성"
cat > backup_important.sh << 'EOF'
#!/bin/bash
# 중요 파일 백업 스크립트

# 백업 디렉토리 생성
backup_dir="$HOME/rl-swarm-backup-$(date +%Y%m%d%H%M%S)"
mkdir -p "$backup_dir"

# 중요 파일 백업
echo "중요 파일 백업 중..."
for file in swarm.pem userData.json userApiKey.json; do
    if [ -f "$file" ]; then
        cp "$file" "$backup_dir/"
        echo "✓ $file 백업 완료: $backup_dir/$file"
    else
        echo "⚠ $file 파일을 찾을 수 없습니다."
    fi
done

# 설정 파일 백업
echo "설정 파일 백업 중..."
if [ -d "hivemind_exp/configs/mac" ]; then
    cp -r hivemind_exp/configs/mac "$backup_dir/"
    echo "✓ 설정 파일 백업 완료: $backup_dir/mac"
fi

echo "백업 완료: $backup_dir"
echo "이 백업을 안전한 위치에 보관하세요."
EOF
chmod +x backup_important.sh
print_success "백업 유틸리티 생성: backup_important.sh"

# 원본 run_rl_swarm.sh 스크립트에서 프로세스 종료 부분 수정
print_header "run_rl_swarm.sh 스크립트 최적화"
if [ -f "run_rl_swarm.sh" ]; then
    # 백업 생성
    cp run_rl_swarm.sh run_rl_swarm.sh.backup
    
    # 프로세스 종료 부분을 안전하게 수정
    sed -i '' 's/kill -9 $PID/if [ -n "$PID" ] \&\& ps -p $PID > \/dev\/null 2>\&1; then kill -9 $PID; fi/g' run_rl_swarm.sh 2>/dev/null || {
        print_warning "자동 수정에 실패했습니다. 수동 수정이 필요할 수 있습니다."
    }
    
    print_success "run_rl_swarm.sh 스크립트 최적화 완료"
else
    print_warning "run_rl_swarm.sh 파일을 찾을 수 없습니다."
fi

# 완료 메시지
print_header "설치 및 최적화가 완료되었습니다"
echo "다음 명령으로 RL-Swarm을 최적화된 설정으로 실행할 수 있습니다:"
echo "  cd ~/rl-swarm && ./run_optimized.sh"
echo ""
echo "RL-Swarm을 지금 실행하시겠습니까? [Y/n]"
read -r run_choice

if [[ ! "$run_choice" =~ ^[Nn]$ ]]; then
    print_header "RL-Swarm 실행 중..."
    ./run_optimized.sh
fi
