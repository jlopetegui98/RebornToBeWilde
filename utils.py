# Description: This file contains useful methods for the project
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

# function to read the texts of an specific author
def read_texts(path: str, label, len_to_read =None, max_length = 350):
    """
    Read the texts of an specific author and return a dictionary with the texts and the labels
    inputs:
      path: str: path to the file with the texts
      label: int: label to assign to the texts
      len_to_read: int: number of texts to read from the author
      max_length: int: max length of the texts to return
    outputs:
      dt: dict: dictionary with the texts and the labels
    """
    text = ''
    with open(path, 'r+', encoding='utf-8') as fd:
      text = fd.read()
      if len_to_read != None:
        text = text[:len_to_read]
    text_splited = text.split()
    dt = {'text': [], 'label': []}
    for i in range(0,len(text_splited),max_length):
      text = ' '.join(text_splited[i:min(i+max_length, len(text_splited))])
      dt['text'].append(text)
      dt['label'].append(label)
    return dt


# function to load model
def load_model(model_name, adapt = False, from_finetuned = False, model_path = None):
    """
    Load a model from the models folder
    inputs:
        model_name: str: name of the model to load
        adapt: bool: if the model is going to be adapted
        from_finetuned: bool: if the model is going to be loaded from a fine-tuned model
        model_path: str: path to the fine-tuned model
    outputs:
        model: model: model loaded
    """
    bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
    )
    model = AutoModelForCausalLM.from_pretrained(
       model_name,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    model.config.use_cache = False # silence the warnings. Please re-enable for inference!
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    #Adding the adapters in the layers in case of adaptation
    if adapt:
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
            )
        model = get_peft_model(model, peft_config)
    # Load the model from the fine-tuned model state dict
    if from_finetuned:
        model.load_state_dict(torch.load(model_path))
    return model
    
    
def load_tokenizer(model_name):
    """
    Load a tokenizer from the models folder
    inputs:
      model_name: str: name of the model to load
    outputs:
        tokenizer: tokenizer: tokenizer loaded
    """    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token, tokenizer.add_eos_token
    return tokenizer

# function to tokenize the input in the expected form of the prompt
def tokenize(tokenizer, text):
    """
    Tokenize the input in the expected form of the prompt
    inputs:
      tokenizer: tokenizer: tokenizer to use
      text: str: text to tokenize without the prompt tokens
    outputs:
      tokenizer: tokenizer: tokenizer to use
    """
    return tokenizer(f"<s>[INST]This are the first lines of a work of fiction. Continue it. {text} [/INST]", return_tensors = "pt", add_special_tokens = False)

# main function for experiments
def clf_exp(model, tokenizer, clf, texts):
    """
    Function for the experiments: Given a model, a tokenizer and a classifier, it generates text using the 
    lines in the texts list and then predicts the label of the generated text using the classifier
    inputs:
      model: model: model to use
      tokenizer: tokenizer: tokenizer to use
      clf: classifier: classifier to use
      texts: list: list of strings with the lines to generate and predict
    outputs:
        label_predictions: list: list of predictions of the classifier
        generated_texts: list: list of the generated texts
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # pattern to remove the prompt tokens after generation
    patt = r'\[INST]|\[\/INST]|\<s>|\</s>|This are the first lines of a work of fiction. Continue it.'

    generated_texts = [] # list of generated texts
    label_predictions = [] # list of predictions of the classifier
    for input in tqdm(texts):
      tokens = tokenize(tokenizer, input)
      model_inputs = tokens.to(device)
      generated_ids = model.generate(**model_inputs, max_new_tokens=500, do_sample=True)
      decoded = tokenizer.batch_decode(generated_ids)
      decoded = [re.sub(patt, '', x) for x in decoded] # clean the generated texts
      preds, _ = clf.predict(decoded) # predict the label of the generated text with clf
      label_predictions.extend(preds)
      generated_texts.extend(decoded)
      del model_inputs
      del decoded
      del generated_ids
    return label_predictions, generated_texts

# function to save the generated texts and labels
def save_generated_texts_and_labels(texts, labels, model = 'baseline', data_path = 'data'):
    """
    Save the generated texts and labels in a json file
    inputs:
      texts: list: list of generated texts
      labels: list: list of labels of the generated texts
      model: str: name of the model used to generate the texts
    """
    dict_text_to_author = {'text': [], 'label': []}

    for i in range(len(texts)):
      dict_text_to_author['text'].append(texts[i])
      dict_text_to_author['label'].append(str(labels[i]))

    with open(f"{data_path}/{model}_generated_texts.json", 'w+') as fd:
      json.dump(dict_text_to_author, fd)