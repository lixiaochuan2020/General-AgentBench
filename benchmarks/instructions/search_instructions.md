Follow the `README.md` in the `deepresearch_llm_modeling` directory to get your environment set up. Make sure to preprare the data in the specific format.

If you are using Bedrock, you need to change the model name in the `main_parallel.py` script to reflect the model you want to use. You will have to export `AWS_BEARER_TOKEN_BEDROCK` and `SERPER_API_KEY` if you want to use Bedrock.

Here is an example of how you can run the script:
```bash
python main_parallel.py --batch_file evaluation/report/mind2web2/mind2web2_16k.json  \
    --search_engine serper \
    --log_dir mind2web2/deepseek_r1 \
    --answer_dir mind2web2/deepseek_r1_results \
    --is_bedrock \
    --use_explicit_thinking
```

Here is an example of running the script while using a target context length:
```bash
python main_parallel.py --batch_file evaluation/report/consolidated/stratified_sample_10pct.json \
    --search_engine serper \
    --log_dir 10pct/qwen3_64k \
    --answer_dir 10pct/qwen3_64k_results \
    --is_bedrock \
    --use_explicit_thinking \
    --target_context_length 64000
```
