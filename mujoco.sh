env=$1
total_timesteps=${2:-3000000}
tt=3000000   # Change this for specific envs
if [ $# -le 0 ]; then
    echo './mujoco.sh <env> <timesteps>'
    exit
fi


for seed in {1,2,4,}
do
    python main.py --env-name ${env}-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 8 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps ${tt} --use-linear-lr-decay --use-proper-time-limits --gail --model_name ${env}BCGAIL --seed ${seed} --bcgail 1 --gailgamma 1 --decay 10 &
    #python main.py --env-name ${env}-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 8 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps $total_timesteps --use-linear-lr-decay --use-proper-time-limits --gail --model_name ${env}GAIL --seed ${seed}
done
