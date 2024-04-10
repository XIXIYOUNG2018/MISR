[ -z "${exp_name}" ] && exp_name="8000"
[ -z "${seed}" ] && seed="1"
# [ -z "${arch}" ] && arch="--ffn_dim 768 --hidden_dim 768 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 5"
[ -z "${arch}" ] && arch="--ffn_dim 1024 --hidden_dim 1024 --dropout_rate 0.1 --attention_dropout_rate 0.3 --n_layers 18 --peak_lr 1e-4 --edge_type multi_hop --multi_hop_max_dist 5"
[ -z "${batch_size}" ] && batch_size="128"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "batch_size: ${batch_size}"
echo "==============================================================================="

default_root_dir="../../exps/admet/$exp_name/$seed"
mkdir -p $default_root_dir
n_gpu=$(nvidia-smi -L | wc -l)
max_epochs=20
# datasets=(SR-MMP)
# datasets=(SR-p53)
# datasets=(NR-PPAR-gamma  SR-HSE)
# datasets=(SR-ATAD5 SR-ARE)
# datasets=(NR-ER NR-AR-LBD)
# datasets=(NR-Aromatase)
# datasets=(ROA )
# datasets=(NR-AhR EC)
# datasets=(NR-AR NR-ER-LBD)

# datasets=(F30%  F20%)
# datasets=(HIA FDAMDD)
# datasets=(EI)
# datasets=(Pgp-sub Pgp-inh)
# datasets=(CYP2C9-sub CYP1A2-inh)
# datasets=(SkinSen CYP2D6-inh CYP2D6-sub)
# datasets=(DILI Ames)
# datasets=( CYP2C19-sub DILI)
# datasets=(CYP2C19-inh Ames CYP2C19-sub DILI)
# datasets=(T12 BBB)
# datasets=( hERG Carcinogenicity)
# datasets=(CYP1A2-sub Respiratory)
# datasets= ( H-HT CYP3A4-inh )
# CYP3A4-inh CYP2C9-inh)
# datasets=(CYP2C9-inh  CYP2C9-sub)
# datasets=(CYP3A4-inh  CYP3A4-sub)
# datasets=(CYP2C19-inh  CYP2C19-sub)
# datasets=(CYP1A2-inh CYP1A2-sub)
# datasets=(LogS LogD LogP MDCK PPB VDss Fu CL Caco-2 IGC50 LC50DM LC50 BCF)
# for element in ${datasets[@]}
# do
# echo $element
python ../../graphormer/entry.py --num_workers 4 --seed $seed --batch_size $batch_size \
      --dataset_name LogD_smi    --checkpoint_path /home/ps/Documents/xxy/pred/Admethormer_finetune/exps/pretrain/mask/last.ckpt\
      --gpus $n_gpu --accelerator cuda --precision 16 --gradient_clip_val 5.0 \
      $arch \
      --default_root_dir $default_root_dir --max_epochs $max_epochs  
# done

