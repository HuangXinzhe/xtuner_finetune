# xtuner_finetune
通过xtuner微调模型的步骤：
1. 安装xtuner
2. 制作数据集
3. 微调
4. 将微调后的模型转换为HuggingFace模型
5. 可以直接使用，或是上传至HuggingFace Hub

## 1. 安装xtuner
```
# 创建版本文件夹并进入
mkdir xtuner019 && cd xtuner019

# 拉取 0.1.9 的版本源码
git clone -b v0.1.9  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
# git clone -b v0.1.9 https://gitee.com/Internlm/xtuner

# 进入源码目录
cd xtuner

# 从源码安装 XTuner
pip install -e '.[all]'
```

## 2. 制作数据集
```
# 创建存放数据集的文件夹
mkdir ~/ft-oasst1 && cd ~/ft-oasst1

# 微调所需的数据与所使用的模型有关，根据需要制作相应的数据集
# 此次是微调InternLM-chat-7b模型，数据集为对话数据集，格式为：
{"text": "### Human: xxx.### Assistant: xxx.### Human: xxx."}
# https://huggingface.co/datasets/timdettmers/openassistant-guanaco
# 将数据集加载到ft-oasst1文件夹下
```

## 3. 微调

模型下载：可以下载微调所需的模型，此次使用的是InternLM-chat-7b模型
注意：需要修改pretrained_model_name_or_path与data_path中的地址
模型下载至ft-oasst1

XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：

```
# 列出所有内置配置
xtuner list-cfg

baichuan2_13b_base_qlora_alpaca_e3
# 模型名称_模型参数量_模型类型_微调方式_微调数据集_训练轮数

# 在创建的数据集文件夹ft-oasst1下拷贝配置文件到当前目录
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
# xtuner copy-cfg 所选配置文件名 要复制到的位置
```

开始微调
```
# 单卡
## 用刚才改好的config文件训练
xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py

# 多卡
NPROC_PER_NODE=${GPU_NUM} xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py

# 若要开启 deepspeed 加速，增加 --deepspeed deepspeed_zero2 即可
```

tmux new -s finetune
ctrl+b d退出
tmux attach -t finetune
回到之前的tmux
退出之后tmux中的代码也依旧在运行，还可以重新进入tmux查看代码运行情况


## 4. 将微调后的模型转换为HuggingFace模型
将得到的 PTH 模型转换为 HuggingFace 模型，即：生成 Adapter 文件夹
```
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1

xtuner convert pth_to_hf ./internlm_chat_7b_qlora_oasst1_e3_copy.py ./work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth ./hf
```
adapter_model.safetensors这是HuggingFace的lora模型

## 部署测试
将 HuggingFace adapter 合并到大语言模型
```
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
xtuner convert merge 基座模型路径 HuggingFace adapter 路径 合并后的模型路径 模型每个分片的最大大小
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB
```

## 与合并后的模型对话
```
# 加载 Adapter 模型对话（Float 16）
xtuner chat ./merged --prompt-template internlm_chat

# 4 bit 量化加载
# xtuner chat ./merged --bits 4 --prompt-template internlm_chat
```
不同的模型对应的prompt-template不同，可以通过查看如下命令找到对应的prompt-template
xtuner chat --help
对比微调前后的效果可以通过该模型文件地址，以上为例
xtuner chat ./internlm-chat-7b --prompt-template internlm_chat

## Demo
cli_demo.py中修改model_name_or_path
运行cli_demo.py


### 注：详细内容见https://github.com/InternLM/tutorial/blob/main/xtuner/README.md