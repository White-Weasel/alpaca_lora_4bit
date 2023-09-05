import time
import torch
from alpaca_lora_4bit.autograd_4bit import load_llama_model_4bit_low_ram, load_llama_model_4bit_low_ram_and_offload, Autograd4bitQuantLinear
from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import replace_peft_model_with_int4_lora_model
replace_peft_model_with_int4_lora_model()

config_path = '/tmp/text-generation-webui/models/Neko-Institute-of-Science_LLaMA-13B-4bit-128g/'
model_path = '/tmp/text-generation-webui/models/Neko-Institute-of-Science_LLaMA-13B-4bit-128g/llama-13b-4bit-128g.safetensors'
lora_path = '/content/gdrive/MyDrive/ColabNotebooks/Lora/Salie/adapter_model.bin'

# For test lora, Use this:
model, tokenizer = load_llama_model_4bit_low_ram_and_offload(config_path, model_path, lora_path=lora_path,
                                                             groupsize=128, seqlen=2048, bits=4,
                                                             max_memory={0: '15Gib', 'cpu': '12Gib'},
                                                             is_v1_model=True)

# model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path, groupsize=-1)

print('Fitting 4bit scales and zeros to half')
model.half()
for n, m in model.named_modules():
    if isinstance(m, Autograd4bitQuantLinear):
        if m.is_v1_model:
            m.zeros = m.zeros.half()
        m.scales = m.scales.half()
        m.bias = m.bias.half()

print('Apply AMP Wrapper ...')
from amp_wrapper import AMPWrapper
wrapper = AMPWrapper(model)
wrapper.apply_generate()

prompt = '''As Salie, response to the following conversation.

### Conversation:
@Westsidebill: Much better, sleeping better, more energy. I did gain about 15 lbs.
@durbblurb:  I did gain about 15 lbs.
That part is not always mentioned.
@sleeps_too_little: Smoking used to be advertised as a weight loss method apparently
@lionx0x: Makes sense. I dropped 30 pounds in a few months last summer from smoking. Looked great felt like shit though lol.
@Salie:

### Response:
'''
batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
batch = {k: v.cuda() for k, v in batch.items()}

start = time.time()
with torch.no_grad():
    generated = model.generate(inputs=batch["input_ids"],
                               do_sample=True, use_cache=True,
                               repetition_penalty=1.1,
                               max_new_tokens=500,
                               temperature=0.9,
                               top_p=0.95,
                               top_k=40,
                               return_dict_in_generate=True,
                               output_attentions=False,
                               output_hidden_states=False,
                               output_scores=False)
result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
end = time.time()
print(result_text)
print(end - start)
