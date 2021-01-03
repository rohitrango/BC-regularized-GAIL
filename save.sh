env=$1
savedir=$2
if [ $# -ne 2 ]; then
    echo './save.sh <env> <savedir>'
    exit
fi

for seed in {1,2,4}
do
    python main.py --env-name ${env} --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --model_name demo --load_model_name ${savedir} --record_trajectories 1 --seed ${seed}
    ## Save random policy
    #python main.py --env-name ${env} --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --model_name random --load_model_name ${savedir} --record_trajectories 1 --seed ${seed}
done
