mkdir data/
mkdir log/
mkdir log_DA/
mkdir best_p_domain_mixture/

# training and testing FacLens on each individual domain
# use the hidden states of the last input token as the hidden question represention
for model in llama llama3 mistral
do
    for data in PopQA EQ NQ
    do
        for layer_name in middle_layer second_to_last_layer last_layer
        do
            for learning_rate in 1e-3 1e-4
            do
                python train.py --llm $model --dataset $data --layer_name $layer_name --learning_rate $learning_rate --use_last_token
            done
        done
    done
done

# use the averaged hidden states of a question's tokens as its hidden question representation
for model in llama llama3 mistral
do
    for data in PopQA EQ NQ
    do
        for layer_name in middle_layer second_to_last_layer last_layer
        do
            for learning_rate in 1e-3 1e-4
            do
                python train.py --llm $model --dataset $data --layer_name $layer_name --learning_rate $learning_rate
            done
        done
    done
done

# use the averaged hidden states of input entites' tokens as the hidden question representation
for model in llama llama3 mistral
do
    for data in PopQA EQ NQ
    do
        for layer_name in middle_layer second_to_last_layer last_layer
        do
            for learning_rate in 1e-3 1e-4
            do
                python train.py --llm $model --dataset $data --layer_name $layer_name --learning_rate $learning_rate --use_entity
            done
        done
    done
done

# multi-domain joint training
for data in PopQA EQ NQ
do
    for learning_rate in 1e-3 1e-4
    do
        python train_multi_task.py --dataset $data --layer_name middle_layer --learning_rate $learning_rate
    done
done

# direct transfer across multiple domains
for data in PopQA EQ
do
    for learning_rate in 1e-3
    do
        for llm in llama llama3 mistral
        do
            for llm2 in llama llama3 mistral
            do
                python train_transfer_direct.py --learning_rate $learning_rate --llm $llm --dataset $data --llm2 $llm2 --dataset2 $data
            done
        done
    done
done

for learning_rate in 1e-4
do
    for llm in llama llama3 mistral
    do
        for llm2 in llama llama3 mistral
        do
            python train_transfer_direct.py --learning_rate $learning_rate --llm $llm --dataset NQ --llm2 $llm2 --dataset2 NQ
        done
    done
done

# transfer across multiple domains via domain adaptation
# using linear kernel
for data in PopQA EQ NQ
do
    for learning_rate in 1e-5
    do
        for llm in llama llama3 mistral
        do
            for llm2 in llama llama3 mistral
            do
                python train_transfer.py --learning_rate $learning_rate --epochs 1000 --llm $llm --dataset $data --llm2 $llm2 --dataset2 $data --batch_size 64 --kernel linear
            done
        done
    done
done

# using Gaussion kernel
for data in PopQA EQ NQ
do
    for learning_rate in 1e-5
    do
        for llm in llama llama3 mistral
        do
            for llm2 in llama llama3 mistral
            do
                python train_transfer.py --learning_rate $learning_rate --epochs 1000 --llm $llm --dataset $data --llm2 $llm2 --dataset2 $data --batch_size 64 --kernel rbf
            done
        done
    done
done
