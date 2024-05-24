cd ../
mkdir hidden_states/
cd src_hidden_states/

for llm in llama llama3 mistral
do
    for dataset in PopQA EQ NQ
    do
        python state_last_token.py --model_name $llm --dataset_name $dataset
    done
done

for llm in llama llama3 mistral
do
    for dataset in PopQA EQ NQ
    do
        python state.py --model_name $llm --dataset_name $dataset
    done
done
