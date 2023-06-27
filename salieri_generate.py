import sys
import torch
import fire
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import SaliePrompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_8bit: bool = True,
        base_model: str = "decapoda-research/llama-7b-hf",
        lora_weights: str = "tloen/alpaca-lora-7b",
        prompt_template: str = "salieri",  # The prompt template to use, will default to alpaca.
):
    prompter = SaliePrompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        # fixme: device_map="auto" cause accelerate not using full GPU
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 0
    model.config.eos_token_id = 1

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            message,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=500,
            **kwargs,
    ):
        prompt = prompter.generate_prompt(input=message)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output
        # print(f"output: {output}")
        # return prompter.get_response(output)

    input_ = "@LuckyDemon666: If you want to keep wearing a mask afterwards, you can do that! Personally, I dislike " \
             "masks because they fog up my glasses and make it difficult for me to see. But you do " \
             "you!\n@stanman1979: I wonder though about laws. Some states or counties in the US have laws against " \
             "wearing masks unless you are doing a job that requires them. I suppose public or personal health may " \
             "override, but I'm not sure."
    print("Input:", input_)
    print(f"Response: {evaluate(message=input_)}")


if __name__ == '__main__':
    fire.Fire(main)
    # main(lora_weights="binhgiangnguyendanh/Salieri-Alpaca-Lora-7B")
