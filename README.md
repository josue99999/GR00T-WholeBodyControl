# GR00T-WholeBodyControl — Fine-tuning con Danza del Caporal

Este repositorio es un fork de [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) con el objetivo de hacer fine-tuning del modelo base **GEAR-SONIC** para que el robot humanoide Unitree G1 ejecute la **Danza del Caporal**, una danza folclórica latinoamericana.

---

## ¿Qué es GEAR-SONIC?

GEAR-SONIC es un **foundation model** de control de cuerpo completo para robots humanoides desarrollado por NVIDIA. Funciona como un modelo base generalista entrenado con más de 130,000 secuencias de movimiento humano retargetadas al robot G1. Su arquitectura usa:

- **Encoders múltiples** (G1, SMPL, Teleop) que mapean distintos tipos de entrada a un espacio latente compartido de 64 dimensiones usando FSQ (Finite Scalar Quantization)
- **Un único decoder** que produce comandos de 29 DOF (grados de libertad) para el robot
- **Entrenamiento PPO** en Isaac Lab con miles de entornos paralelos en simulación

Al ser un foundation model, puede recibir nuevas secuencias de movimiento sin reentrenamiento completo — basta con un **fine-tuning** sobre el checkpoint base para que aprenda a ejecutar movimientos específicos con mayor fidelidad.

---

## ¿Qué hace este fork?

1. **Convierte el movimiento de video a datos de entrenamiento**: a partir de un video de la Danza del Caporal procesado con `video2robot`, se extrae la trayectoria del robot en formato PKL y se convierte al formato que espera el sistema de entrenamiento de GEAR-SONIC.

2. **Fine-tuning del foundation model**: se parte del checkpoint publicado por NVIDIA y se continúa el entrenamiento incluyendo la danza caporal como nueva secuencia de movimiento, mejorando la capacidad del modelo para ejecutar ese baile específico sin perder sus capacidades generales.

3. **Deploy en simulación y robot real**: el modelo fine-tuneado se exporta a ONNX y se carga en el sistema de deploy del G1.

---

## Estructura relevante de este fork

```
convert_video2robot_motion.py          # Convierte PKL de video2robot → CSVs de deploy (50 Hz)
setup_training_server.sh               # Setup completo del servidor de entrenamiento
data/motion_lib_custom/robot/
  danza_caporal/danza_caporal.pkl      # Motion en formato motion_lib para training
gear_sonic_deploy/reference/
  danza_caporal/                       # CSVs del movimiento (body_pos, joint_pos, etc.)
  example/danza_caporal/               # Copia para el deploy interactivo
```

---

## Pipeline completo

### 1. Obtener el movimiento desde video

El movimiento de la danza fue procesado con [video2robot](https://github.com/NVlabs/video2robot), que extrae una trayectoria del robot G1 a partir de un video humano. El resultado es un PKL con:

```
fps, root_pos (N,3), root_rot (N,4) [x,y,z,w],
dof_pos (N,29) [orden MuJoCo], local_body_pos (N,38,3)
```

### 2. Convertir a formato de deploy (CSVs a 50 Hz)

```bash
python convert_video2robot_motion.py \
    /ruta/al/robot_motion_g1.pkl \
    gear_sonic_deploy/reference \
    --name danza_caporal
```

Esto genera `joint_pos.csv`, `joint_vel.csv`, `body_pos.csv`, `body_quat.csv`, `body_lin_vel.csv`, `body_ang_vel.csv` en orden IsaacLab a 50 Hz.

### 3. Convertir a formato de entrenamiento (motion_lib PKL)

```bash
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input gear_sonic_deploy/reference/danza_caporal/danza_caporal \
    --output data/motion_lib_custom/robot/danza_caporal/danza_caporal.pkl \
    --fps 50
```

### 4. Probar en simulación (sin fine-tuning)

```bash
# Terminal 1 — simulador MuJoCo
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2 — controlador
cd gear_sonic_deploy
bash deploy.sh sim
# Presiona ] para activar el control
# Presiona Tab para cambiar entre motions (incluye danza_caporal)
```

---

## Fine-tuning en servidor

### Setup del servidor (una sola vez)

Clona este repo en el servidor y corre el script de setup. Instala automáticamente Isaac Sim, Isaac Lab, todas las dependencias y descarga el checkpoint base + datos SMPL desde Hugging Face (~12 GB).

```bash
git clone --branch danza-caporal-finetuning \
    https://github.com/josue99999/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
bash setup_training_server.sh
```

El script hace:
1. Verifica GPU (driver >= 525) y git-lfs
2. Detecta si el disco es NTFS — si es así, crea una imagen ext4 de 30 GB en el mismo disco (pip no puede instalar en NTFS por restricciones POSIX)
3. Instala uv + Python 3.10 (Isaac Sim 4.x solo tiene wheels cp310, no 3.11)
4. Instala Isaac Sim 4.5.0.0 headless via pip (~8-15 GB)
5. Instala Isaac Lab 2.3.0
6. Instala `gear_sonic[training]` (Hydra, TRL, HuggingFace, W&B, etc.)
7. Descarga `sonic_release/last.pt` + `data/smpl_filtered/` desde HuggingFace
8. Genera `activate_training.sh` con la ruta correcta al venv

### Correr el fine-tuning

```bash
source activate_training.sh

# 1 GPU (lento pero funcional)
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=64 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_custom/robot \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered

# 8 GPUs (recomendado)
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_custom/robot \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered

# Con W&B logging
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    +opt=wandb \
    ++wandb.wandb_project=gr00t-danza-caporal \
    ++wandb.wandb_entity=TU_USUARIO_WANDB \
    num_envs=64 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_custom/robot \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
```

Los checkpoints se guardan en `logs_rl/TRL_G1_Track/<exp>-<timestamp>/model_step_XXXXX.pt` cada 2000 steps.

### Exportar a ONNX para deploy

```bash
python gear_sonic/eval_agent_trl.py \
    +checkpoint=logs_rl/TRL_G1_Track/<exp>/model_step_XXXXX.pt \
    +headless=True \
    ++num_envs=1 \
    +export_onnx_only=true
```

Genera los archivos ONNX en `exported/` listos para copiar a `gear_sonic_deploy/policy/`.

---

## Requisitos

| Componente | Requisito |
|---|---|
| GPU | NVIDIA con CUDA 12.x (A100/H100 recomendado, mínimo 1 GPU) |
| Driver | >= 525 |
| Python | 3.10 (training + simulación + deploy) |
| Isaac Lab | 2.3+ |
| Disco | >= 80 GB libres |
| RAM | >= 32 GB |

---

## Créditos

- Modelo base: [NVIDIA GEAR Lab — GEAR-SONIC](https://nvlabs.github.io/GEAR-SONIC/)
- Paper: [GR00T-WBC: A Unified Foundation Model for Humanoid Whole-Body Control](https://nvlabs.github.io/GR00T-WholeBodyControl/)
- Movimiento: Danza del Caporal procesada con video2robot
