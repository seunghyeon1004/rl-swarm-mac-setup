# Apple Silicon Mac용 RL-Swarm 최적화 도구

이 리포지토리는 Apple Silicon(M1/M2/M3/M4) Mac에서 [RL-Swarm](https://github.com/gensyn-ai/rl-swarm)을 최적의 성능으로 실행하기 위한 설정과 스크립트를 제공합니다.

## 주요 기능

- **자동 하드웨어 감지**: 사용 중인 Mac 모델을 자동으로 감지하여 최적의 설정 적용
- **메모리 최적화**: Apple Silicon의 MPS 백엔드에 최적화된 메모리 관리
- **양자화 및 LoRA 설정**: 성능과 효율성을 위한 4비트 양자화 및 LoRA 설정
- **안전 우선**: 중요 파일 백업 및 보존
- **편의 기능**: 원클릭 실행 및 설정 스크립트

## 빠른 시작

```bash
# 한 줄 설치 및 최적화
curl -s https://raw.githubusercontent.com/seunghyeon1004/rl-swarm-mac-setup/main/setup_rl_swarm.sh | bash
