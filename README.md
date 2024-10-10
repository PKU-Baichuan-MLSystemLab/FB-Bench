# How to Use This Project
## 0. Install Required Libraries
Run `pip install -r requirements.txt`, we also recommend you to install vllm for deploying models.

## 1. Fill in the Configuration File
- `config/api_config.yaml`: Set up the API configurations for all models in this file, refer to the comments at the top of the file for specific settings. Open-source models are generally deployed with vllm.
- `config/gen_answer_config.yaml`: This is the configuration file for generating the second round of answers to be evaluated by the model. Refer to the comments in the file for specific settings, usually, only the model_list needs to be modified.
- `config/gen_judgement.yaml`: This is the configuration file for using GPT to evaluate the models. Refer to the comments in the file for specific settings, usually, only the model_list needs to be modified.

## 2. Generate the Second Round Replies of the Models to be Tested
The evaluation dataset is located at `data/feedback-benchmark/fb_bench_dataset.json`.

Run `python gen_answer.py`, the results are saved in `data/feedback-benchmark/model_answer`

## 3. Generate GPT's Evaluation of the Second Round Replies from the Models to be Tested
Run `python gen_judgment.py`, the results are saved in `data/feedback-benchmark/model_judgment`

## 4. Output the Final Score
Run `python show_results.py`


