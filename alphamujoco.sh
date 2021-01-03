env=$1
total_timesteps=${2:-3000000}
if [ $# -le 0 ]; then
    echo './mujoco.sh <env> <timesteps>'
    exit
fi


for seed in {1,2,4,}
do
    #for gamma in {0.5,0.25,0.125}
    for gamma in {0.75,0.50,0.25}
    do
        python main.py --env-name ${env}-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 8 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps ${total_timesteps} --use-linear-lr-decay --use-proper-time-limits --gail --model_name ${env}alphBCGAIL${gamma} --seed ${seed} --bcgail 1 --gailgamma $gamma --decay 1 --num-traj 1 &
    done
    python main.py --env-name ${env}-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 8 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps ${total_timesteps} --use-linear-lr-decay --use-proper-time-limits --gail --model_name ${env}alphBCGAIL --seed ${seed} --bcgail 1 --gailgamma 1 --decay 10 --num-traj 1
done
