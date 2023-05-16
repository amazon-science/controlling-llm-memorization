# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

# delete old logs
rm -r ../logs
mkdir ../logs

<<comment
EXPERIMENT 1
Training soft-prompt by basic language modeling 
comment

echo 'Running baselines'

for i in {1..5}
do
    accelerate launch baseline.py --model_size=small 
    accelerate launch baseline.py --model_size=medium
done

aligned=0
echo 'Running prompt-learning trained with basic language modeling!'
for len_prompt in 1 5 20 100 150
do
    for i in {1..5}
    do
        accelerate launch promptLearn_attack.py --model_size=small --aligned=$aligned --len_prompt=$len_prompt
        accelerate launch promptLearn_attack.py --model_size=medium  --aligned=$aligned --len_prompt=$len_prompt
    done
done


<<comment
EXPERIMENT 2
Training soft-prompt by informed language modeling 
comment

# echo 'Running prompt-learning trained with aligned language modeling!'
# for len_prompt in 1 5 20 100 150
# do
#     for i in {1..5}
#     do
#         accelerate launch promptLearn_attack.py --model_size=small  --len_prompt=$len_prompt
#         accelerate launch promptLearn_attack.py --model_size=medium  --len_prompt=$len_prompt
#     done
# done


<<comment
EXPERIMENT 3
Showing how results scale over different suffix sizes
comment

# echo 'Running baselines over suffix size!'
# for suffix_size in 5 10 25 40
# do
#     for i in {1..5}
#     do
#         accelerate launch baseline.py --model_size=small --suffix_size=$suffix_size
#         accelerate launch baseline.py --model_size=medium --suffix_size=$suffix_size
#     done
# done

# echo 'Running prompt-learning over suffix size!'
# for len_prompt in 1 5 20 100 150
# do
#     for suffix_size in 5 10 25 40
#     do
#         for i in {1..5}
#         do
#             accelerate launch promptLearn_attack.py --model_size=small  --len_prompt=$len_prompt --suffix_size=$suffix_size
#             accelerate launch promptLearn_attack.py --model_size=medium   --len_prompt=$len_prompt --suffix_size=$suffix_size
#         done
#     done
# done


<<comment
EXPERIMENT 4
Showing how results scale over prefix/context size
comment

# echo 'Running baselines over prefix size!'
# for prefix_size in 25 75 100 125
# do
#     for i in {1..5}
#     do
#         accelerate launch baseline.py --model_size=small --prefix_size=$prefix_size
#         accelerate launch baseline.py --model_size=medium --prefix_size=$prefix_size
#     done
# done

# echo 'Running prompt-learning over different prefix/context size!'
# for len_prompt in 1 5 20 100 150
# do
#     for prefix_size in 25 75 100 125
#     do
#         for i in {1..5}
#         do
#             accelerate launch promptLearn_attack.py --model_size=small  --len_prompt=$len_prompt --prefix_size=$prefix_size
#             accelerate launch promptLearn_attack.py --model_size=medium   --len_prompt=$len_prompt --prefix_size=$prefix_size
#         done
#     done
# done

<<comment
EXPERIMENT 5
Showing how results scale over beam decoding
comment

# echo 'Running baselines over beam decoding!'
# for num_beams in 5 10 15 20
# do
#     for i in {1..5}
#     do
#         accelerate launch baseline.py --model_size=small --num_beams=$num_beams
#         accelerate launch baseline.py --model_size=medium --num_beams=$num_beams
#     done
# done

# echo 'Running prompt-learning over beam decoding!'
# for len_prompt in 1 5 20 100 150
# do
#     for num_beams in 5 10 15 20
#     do
#         for i in {1..5}
#         do
#             accelerate launch promptLearn_attack.py --model_size=small  --len_prompt=$len_prompt --num_beams=$num_beams
#             accelerate launch promptLearn_attack.py --model_size=medium   --len_prompt=$len_prompt --num_beams=$num_beams
#         done
#     done
# done


<<comment
EXPERIMENT 6
Defense experiments ()
comment

# echo 'Running GPT2 variants for defense!'
# for i in {1..5}
# do
#     accelerate launch baseline.py --model_size=gpt2
#     accelerate launch baseline.py --model_size=gpt2XL
# done


#echo 'Running prompt-(un)learning for defense!'
# len_prompt=1
# for theta in 1.25 1.5 1.75
# do
#     for i in {1..5}
#     do
#         accelerate launch promptLearn_defense.py --model_size=small  --len_prompt=$len_prompt --theta=$theta
#     done
# done

# for theta in 0.5 0.75 1.0
# do
#     for i in {1..5}
#     do
#         accelerate launch promptLearn_defense.py --model_size=medium  --len_prompt=$len_prompt --theta=$theta
#     done
# done


