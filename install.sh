git clone --depth=1 --branch main https://github.com/oobabooga/text-generation-webui.git text-generation-webui-tmp
(
cd text-generation-webui-tmp || exit
printf '%s'"import custom_monkey_patch # apply monkey patch\n" | cat - server.py > tmpfile && mv tmpfile server.py
python download-model.py Neko-Institute-of-Science/LLaMA-13B-4bit-128g --threads 8
python binhgiangnguyendanh/test_salie_lora --threads 8
mkdir ../models/Neko-Institute-of-Science_LLaMA-13B-4bit-128g/
mkdir ../lora/Salie/
mv models/Neko-Institute-of-Science_LLaMA-13B-4bit-128g/* ../models/Neko-Institute-of-Science_LLaMA-13B-4bit-128g/
mv loras/binhgiangnguyendanh_test_salie_lora/* ../lora/Salie/
)
mv text-generation-webui-tmp/* text-generation-webui/
rm -rf text-generation-webui-tmp
python text-generation-webui/server.py