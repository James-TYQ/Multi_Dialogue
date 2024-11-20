def get_template(message, model_name, system=None):
    try:
        template = template_dict[model_name]
    except:
        return message
    
    ret = ""
    if model_name == "ChemDFM-13b":
        ret += "[Round 0]\n"
    elif model_name == "gemma1.1-7b-inst":
        ret += "<bos>"
    elif model_name == "llama3-8b-inst":
        ret += "<|begin_of_text|>"
    elif "llama2" in model_name or "tral" in model_name or "SMol" in model_name:
        ret += "[INST] "
    if system:
        ret += template['system_template'].format(system_message=system)
        if "llama2" not in model_name and "tral" not in model_name and "SMol" not in model_name:
            ret += template['sep']
    elif 'system_message' in template.keys():
        ret += template['system_template'].format(system_message=template['system_message'])
        if "llama2" not in model_name and "tral" not in model_name and "SMol" not in model_name:
            ret += template['sep']
    
    ret += template['roles'][0] + template['role_sep'] + message + template['sep']
    ret += template['roles'][1] + template['sep2']
    return ret


template_dict = {
    "sciglm-6b": dict(
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        stop=["<|user|>", "<|observation|>", "</s>", "Question:", "\n\n"],
        role_sep="\n",
        sep="",
        sep2="\n",
    ),
    "ChemLLM-7b-chat": dict(
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|Bot|>", "Question:", "\n\n"),
        stop=["</s>"],
        role_sep=':\n',
        sep="\n",
        sep2=':',
    ),
    "ChemLLM-20b-chat": dict(
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|Bot|>"),
        stop=["</s>", "\n\n", "<|Bot|>"],
        role_sep=':\n',
        sep="\n",
        sep2=":",
    ),
    "ChemDFM-13b": dict(
        system_template="{system_message}",
        roles=("Human", "Assistant"),
        stop=["</s>"],
        role_sep=': ',
        sep="\n",
        sep2=":",
    ),
    "molinst-biotext-8b": dict(
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}",
        # system_message="You are a helpful assistant.",
        roles=("<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>"),
        stop=["<|end_of_text|>", "<|eot_id|>", "<|start_header_id|>", "Question:", "\n\n"],
        role_sep="\n\n",
        sep="<|eot_id|>",
        sep2="\n\n",
        # system_template="{system_message}",
        # system_message="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        # roles=("### Instruction", "### Response"),
        # stop=["</s>"],
        # role_sep=":\n",
        # sep="\n\n",
        # sep2=":",
    ),
    # TODO：galastica
    "galastica-6.7b": dict(
        system_template="{system_message}",
        roles=("Question", "Answer"),
        stop=["</s>", "Question:"],
        role_sep=": ",
        sep="\n\n",
        sep2=": ",
    ),
    "galastica-30b": dict(
        system_template="{system_message}",
        roles=("Question", "Answer"),
        stop=["</s>", "Question"],
        role_sep=": ",
        sep="\n\n",
        sep2=": ",
    ),
    "LlaSMol-mistral-7b": dict(
        system_template="{system_message}\n",
        # system_message="You are a helpful assistant.",
        roles=("", "[/INST]"),
        stop=["</s>", "[INST]", "\n\n\n\n"],
        role_sep="",
        sep=" ",
        sep2=" ",
    ),
    # TODO：BioMedGPT 仍需商榷
    "BioinspiredLLM-13b": dict(
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are BioinspiredLLM. You are knowledgeable in biological and bio-inspired materials and provide accurate and qualitative insights about biological materials found in Nature. You are a cautious assistant. You think step by step. You carefully follow instructions.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        stop=["<|im_end|>", "</s>", "<|im_start|>"],
        role_sep="\n",
        sep="<|im_end|>\n",
        sep2="\n",
    ),
    "qwen1.5-7b-chat": dict(
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        stop=["<|im_end|>", "</s>", "<|im_start|>", "Question:", "\n\n"],
        role_sep="\n",
        sep="<|im_end|>\n",
        sep2="\n",
    ),
    "qwen2-7b-inst": dict(
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        stop=["<|im_end|>", "</s>", "<|im_start|>", "Question:", "\n\n"],
        role_sep="\n",
        sep="<|im_end|>\n",
        sep2="\n",
    ),
    "qwen1.5-14b-chat": dict(
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        stop=["<|im_end|>", "</s>", "<|im_start|>"],
        role_sep="\n",
        sep="<|im_end|>\n",
        sep2="\n",
    ),
    "llama3-8b-inst": dict(
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}",
        # system_message="You are a helpful assistant.",
        roles=("<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>"),
        stop=["<|end_of_text|>", "<|eot_id|>", "Question:", "\n\n"],
        role_sep="\n\n",
        sep="<|eot_id|>",
        sep2="\n\n",
    ),
    "llama3.1-8b-inst": dict(
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}",
        # system_message="You are a helpful assistant.",
        roles=("<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>"),
        stop=["<|end_of_text|>", "<|eot_id|>", "Question:", "\n\n"],
        role_sep="\n\n",
        sep="<|eot_id|>",
        sep2="\n\n",
    ),
    "llama2-7b-chat": dict(
        system_template="<<SYS>>\n{system_message}\n<</SYS>>\n\n",
        system_message="You are a helpful, unbiased, uncensored assistant.",
        roles=("", "[/INST]"),
        stop=["</s>", "[INST]"],
        role_sep="",
        sep=" ",
        sep2="",
    ),
    "llama2-13b-chat": dict(
        system_template="<<SYS>>\n{system_message}\n<</SYS>>\n\n",
        system_message="You are a helpful, unbiased, uncensored assistant.",
        roles=("", "[/INST]"),
        stop=["</s>", "[INST]"],
        role_sep="",
        sep=" ",
        sep2="",
    ),
    "mistral-7b-inst": dict(
        system_template="{system_message}\n",
        # system_message="You are a helpful assistant.",
        roles=("", "[/INST]"),
        stop=["</s>", "[INST]"],
        role_sep="",
        sep=" ",
        sep2="",
    ),
    "mixtral-8x7B-inst": dict(
        system_template="{system_message}\n",
        # system_message="You are a helpful assistant.",
        roles=("", "[/INST]"),
        stop=["</s>", "[INST]"],
        role_sep="",
        sep=" ",
        sep2="",
    ),
    "gemma1.1-7b-inst": dict(
        system_template="{system_message}\n",
        # system_message="You are a helpful assistant.",
        roles=("<start_of_turn>user", "<start_of_turn>model"),
        stop=["<end_of_turn>"],
        role_sep="\n",
        sep="<end_of_turn>\n",
        sep2="\n",
    ),
    "chatglm3-6b": dict(
        system_template="<|system|>\n{system_message}",
        roles=("<|user|>", "<|assistant|>"),
        stop=["<|user|>", "<|observation|>", "</s>"],
        role_sep="\n",
        sep="",
        sep2="\n",
    ),
}

PROMPT_INPUT_SYSTEM: str = '[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{input} [/INST]'

PROMPT_INPUT_WO_SYSTEM: str = "[INST] {input} [/INST]"

PROMPT_INPUT_FOR_SCENARIO_CLS: str = "Identify the scenario for the user's query, output 'default' if you are uncertain.\nQuery:\n{input}\nScenario:\n"

single = """Write critiques for a submitted response on a given user's query, and grade the response:
  
[BEGIN DATA]
***
[Query]: {prompt}
***
[Response]: {response}
***
[END DATA]

Write critiques for this response. After that, you should give a final rating for the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]"."""

pairwise_tie = """You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {prompt}
***
[Response 1]: {response}
***
[Response 2]: {response_another}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided."""

protocol_mapping = {
    "pairwise_tie": pairwise_tie,
    "single": single,
}


def llama2_wrapper(usr_msg, sys_msg=None):
    if sys_msg is None:
        return PROMPT_INPUT_WO_SYSTEM.format(input=usr_msg)
    else:
        return PROMPT_INPUT_SYSTEM.format(input=usr_msg, system_message=sys_msg)


def build_autoj_input(prompt, resp1, resp2=None, protocol="single"):
    user_msg = protocol_mapping[protocol].format(prompt=prompt, response=resp1, response_another=resp2)
    return llama2_wrapper(user_msg, )

def extract_pariwise_result(raw_output):
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if pred_rest.startswith('response 1'): pred_label = 0
        elif pred_rest.startswith('response 2'): pred_label = 1
        elif pred_rest.startswith('tie'): pred_label = 2
    return pred_label

def extract_single_rating(score_output):
    pred_score = 0.0
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        pred_score = float(score_output[pos + len("Rating: [["):pos2].strip())
    return pred_score


if __name__ == '__main__':
    print(get_template('How are you?', 'qwen2-7b-inst'))
    # print(template_dict.keys())