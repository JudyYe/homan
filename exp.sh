python -m make_fig -m \
    expname=default_1_1/\${index} \
    cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar ind=1,2 video=True \
    hydra/launcher=learnlab


python test.py -m \
    expname=default/\${index} \
    cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar ind=1,2 \
    hydra/launcher=learnlab


python test.py -m \
    expname=other_\${coarse.lw_depth}_\${fine.lw_depth}/\${index}_\${obj_index} \
    obj_file=\${environment.data_dir}/HOI4D/\${index}/other_cad/\${obj_index}.obj \
    obj_index=000,001,002,003,004 \
    cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar ind=1,2 \
    coarse.lw_depth=1  \
    +engine=learnlab \

python test.py -m \
    expname=worst_\${coarse.lw_depth}_\${fine.lw_depth}/\${index}_\${obj_index} \
    obj_file=\${environment.data_dir}/HOI4D/\${index}/worst_template/\${obj_index}.obj \
    obj_index=000,001,002,003,004 \
    cat=Mug,Bottle,Kettle,Bowl,Knife,ToyCar ind=1,2 \
    coarse.lw_depth=1  \
    +engine=learnlab \



python test.py -m \
    expname=other_\${coarse.lw_depth}_\${fine.lw_depth}/\${index}_\${obj_index} \
    obj_file=\${environment.data_dir}/HOI4D/\${index}/other_cad/\${obj_index}.obj \
    obj_index=000 \
    cat=Mug \
    coarse.lw_depth=1 \
    logging=none 



python test.py -m \
    expname=worst_\${coarse.lw_depth}_\${fine.lw_depth}/\${index}_\${obj_index} \
    obj_file=\${environment.data_dir}/HOI4D/\${index}/worst_template/\${obj_index}.obj \
    obj_index=000,001,002 \
    cat=Mug \
    coarse.lw_depth=1 \
    logging=none 