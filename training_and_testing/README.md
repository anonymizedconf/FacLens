# FacLens
This is the training and tesing source code of FacLens.

In this work, we aim to explore whether we can utilize a lightweight probe to elicit "whether an LLM knows" (i.e. NFP) from the hidden representations of questions. Through extensive experimental analysis, we demonstrate that the hidden representations of questions indeed contain valuable patterns for NFP tasks, and such patterns are transferable across domains corresponding to different LLMs.

# Overview
Specifically, the repository is organized as follows:

* `final_data/` contains the constructed NFP datasets.

* `src_hidden_states/` contains code for generating hidden question representations.

* `src/` contains code for training and testing FacLens.

# Running the code
```
$ cd src_hidden_states/
$ sh run_states.sh
$ cd ../
$ cd src/
$ sh run.sh
```

Please note that you need to download the checkpoints of LLaMA2-7B-Chat, LLaMA3-8B-Instruct, and Mistral-7B-Instruct-v0.2 before running the code.
Besides, you need to enter **src_hidden_states/** and open the files **state_last_token.py** and **state.py** to modify the **file_prefix**, so that the checkpoints can be loaded.