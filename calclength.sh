for env in {Ant,Hopper,HalfCheetah,Reacher,Walker2d}
do
    #for method in {BC,BCnoGAIL,GAIL,noGAIL}
    for method in {BCGAIL,}
    do
        python main.py --env-name ${env}-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 8 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --record_trajectories 1 --model_name ${env}${method} --load_model_name ${env}${method} --savelength 1
    done
done
