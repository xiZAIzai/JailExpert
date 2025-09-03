EXPERIMENT=main
experience_type=full
experience_name=llama-3
target_model=gpt-oss-20b
strategy=baseline
top_k=1.0
device=cuda:4
targe_api=
attack_api=
eval_api=

nohup python ../codes/attack.py --experiment $EXPERIMENT \
                                     --experience_type $experience_type \
                                     --experience_name $experience_name \
                                     --target_model $target_model \
                                     --strategy $strategy \
                                     --top_k $top_k \
                                     --targe_api $targe_api \
                                     --attack_api $attack_api \
                                     --eval_api $eval_api \
                                     --device $device > ./logs/${EXPERIMENT}_${strategy}_${experience_type}_${target_model}.log 2>&1 &