env=$1
timesteps=${2:-3000000}
if [ $# -le 0 ]; then
    echo './mujoco.sh <env> <timesteps>'
    exit
fi

# Run behavior cloning and pretrain
for seed in {1,2,4,}
do
    python bc.py --env-name ${env}-v2 --algo ppo --use-gae --log-interval 1 --num-steps 1000 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --model_name ${env}BC --gail --save-interval 1 --seed ${seed}
    python main.py --env-name ${env}-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 8 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps ${timesteps} --use-linear-lr-decay --use-proper-time-limits --gail --load_model_name ${env}BC --model_name ${env}GAILpretrain --seed ${seed}
done
