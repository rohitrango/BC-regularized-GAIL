for env in {Hopper,Ant,HalfCheetah,Walker2d,Reacher}
do
    #for method in {BCGAIL,BC,GAIL}
    for method in {RED,SAIL,GAILpretrain}
    do
        ./save.sh ${env}-v2 ${env}${method}
    done
done
