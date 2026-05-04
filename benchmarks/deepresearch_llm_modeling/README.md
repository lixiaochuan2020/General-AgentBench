## Run the Agent

### 1. Data Preparation

Data must have field `'id'` and '`question`'.

### 2. Model Deployment

If you want to use local deployed model like Qwen3, use vllm to deploy your model and change the base_url in following code snippet in `main_parallel.py` to match the url.

```
self.client = OpenAI(
                api_key='EMPTY',
                base_url="address_for_your_vllm"
            )
```

Note: If you are using Qwen3 series mode, please enable the thinking mode. You may also add rope scaling for context issue.

### 3. Run the Agent

#### Main Parameters:

```
--batch_file: Path to the data file (required)
--answer_dir: Directory for result output (default: results)
--log_dir: Directory for log output (default: logs)
--long_report: If added, outputs long reports; otherwise outputs short answers
--is_qwen: Use Qwen model; otherwise defaults to Gemini model
--search_engine: Search engine type (default: clueweb, options: serper, etc.)
--use_explicit_thinking: Use Chain of Thought thinking prompts, suitable for models without internal thinking (e.g., Qwen2.5 series, Qwen3-4B-Instruct)
```

#### Criqiue Parameters (experimental):
```
--use_critique: Use critique mechanism to improve answer quality (disabled by default)
--critique_dir: Directory for storing critique content 
```

#### Run examples

1. Run on Qwen3-1.7B model with cluweb as search engine on WebwalkerQA:
```
python3 main_parallel.py --batch_file evaluation/short/webwalkerqa/test.json \
    --answer_dir results/webwalkerqa/qwen3_1.7b \
    --log_dir logs/webwalkerqa/qwen3_1.7b \
    --is_qwen \
    --search_engine clueweb
```

2. Run on Gemini 2.5 Flash model with serper as search engine on WebwalkerQA:
```
python3 main_parallel.py --batch_file evaluation/short/webwalkerqa/test.json \
    --answer_dir results/webwalkerqa/gemini_2.5_flash \
    --log_dir logs/webwalkerqa/gemini_2.5_flash \
    --search_engine serper
```

3. Run on Qwen3-4B-Instruct with serper as search engine on WebwalkerQA. Enable explicit thinking as this model does not has internal thinking.
```
python3 main_parallel.py --batch_file evaluation/short/webwalkerqa/webwalkerqa_680.json \
    --answer_dir results/webwalkerqa/gemini_2.5_flash \
    --log_dir logs/webwalkerqa/gemini_2.5_flash \
    --search_engine serper \
    --use_explicit_thinking
```

> **Note**: You can replace webwalkerqa with your customize dataset.

## Evaluation

To run evaluation, run:
```
python evaluation/short/evaluation.py \
    --data_path evaluation/short/webwalkerqa/test.json \
    --results_dir results/webwalkerqa/$suffix 

```

> **Note**: You can replace webwalkerqa with your customize dataset.

## SFT
We use Llama-factory for sft.

### 1. Data Preparation (For Distillation)
First, run the teacher agent on the dataset you want, and you will find full log for the input and output of each step under the `log_dir` you set. You may run evaluation and filtering for correct trajecotions.

Then, run the following command to convert the data into the format for SFT.
```
python train/sft/data_reorganize.py
```

Finally, follow the instructions [here](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#data-preparation) to finish other data preparation steps for llama-factory.

### 2. Run SFT

First, set your configuration in a .yaml file.

Then, start SFT by:

```
scripts/train.sh
```

For more additional details, you can refer to the original [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) repo!
