env=$1
total_timesteps=${2:-3000000}
if [ $# -le 0 ]; then
    echo './mujoco.sh <env> <timesteps>'
    exit
fi


for seed in {1,2,4,}
do
    #python main.py --env-name ${env}-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 8 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps ${total_timesteps} --use-linear-lr-decay --use-proper-time-limits --gail --model_name ${env}BCnoGAIL --seed ${seed} --bcgail 1 --gailgamma 0.5 --decay 1 --learn 0 &
    python main.py --env-name ${env}-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 8 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps ${total_timesteps} --use-linear-lr-decay --use-proper-time-limits --gail --model_name ${env}noGAIL --seed ${seed} --learn 0 &
done
