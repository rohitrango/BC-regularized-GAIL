## GAIL baseline
CUDA_VISIBLE_DEVICES=6,7 python main.py --env-name PongNoFrameskip-v4 --algo ppo --gail --gail-experts-dir /serverdata/rohit/BCGAIL/PongPPO/ --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --model_name PongGAIL --gail-batch-size 32 --seed 1
