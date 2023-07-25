python test.py -m \
    expname=default/\${index} \
    cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar ind=1,2 \
    hydra/launcher=learnlab


python test.py -m \
    expname=default_\${coarse.lw_depth}_\${fine.lw_depth}/\${index} \
    cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar ind=1,2 \
    coarse.lw_depth=1,10,100  \
    hydra/launcher=learnlab
