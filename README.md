# Build NanoGPT Plus

This repo is mainly a refactored, modularized, and extended version of the original [BuildNanoGPT](https://github.com/karpathy/build-nanogpt). The BuildNanoGPT project together with [Andrej's step by step video](https://www.youtube.com/watch?v=l8pRSuU81PU) is one of the best learning material even for deep learning researchers. I learned a lot from it. The problem with the original project is that it is not modularized and is not convenient for some follow-up usages and test. So I decided to refactor it and extend it to make it more modular and more easily used and experimented with.

I include model implementations other than just gpt-2. Currently, I added the implementations of Llama, which is done by [hengjiUSTC's learn-llm](https://github.com/hengjiUSTC/learn-llm) with small modifications. I will add more models in the future.

Below is a list of what have been done in this project:

- [x] Decouple model implementation, evaluation, and training. 

The training script is train.py, and the model implementation, and the evaluation with hellaswag, or generation are moved to the corresponding files. Therefore, it is convenient to evaluate a trained model on hellaswag or generate text with the trained model.


- [x] Continue Training.

Now, a partially trained model can be loaded in train.py to continue training it rather than start from scratch all the time.

- [x] Printing Training Progress and Estimated Time.

You had to calculate the training progress and completion time. Now, it is printed.

- [x] Add Model format Convertion.

The trained model now can be converted to the Huggingface transformers format with the function convert_to_hf from convert_to_hf.py.

- [x] Support loading any transformers model.

In the past, only specified pretrained models can be loaded. Now they can all be loaded. The config can be read from transformers model configs.

- [x] Add Comparison Evaluation by an LLM.

Now, we can compare the performance of two LLMs on completion by a well trained LLM with auto_evaluation.py

- [x] Add Training Transformers CausalLM model.

Now, it can train the transformers' CausalLM model in train.py.

- [x] Add Model Loading and Generation.

The function `load_model` is added to load the trained model and generate text with the function `complete` both in utils.py.

- [x] Plot all training logs

Previously, only the loss is plotted. Now, all the training logs that are printed in the command line can be ploted with plot_console_log.py.


TODO:

- [ ] Add support for training with transformers' CausalLM model.
- [ ] Add more models.
- [ ] Add more evaluation datasets.
- [ ] Add more training datasets.

## License

This project is licensed under the terms of the Apache 2.0 License.

## Discord Server

Join our Discord server [here](https://discord.gg/xhcBDEM3).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.