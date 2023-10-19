from ctransformers import AutoModelForCausalLM, AutoConfig, Config
conf = AutoConfig(Config(temperature=0.7, repetition_penalty=1.1, batch_size=52,
                max_new_tokens=1024, context_length=2048))
llm = AutoModelForCausalLM.from_pretrained("./mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                                           model_type="mistral", config = conf)
prompt = "What is Science?"
template = f'''<s>[INST] {prompt} [/INST]'''
print(llm(template))